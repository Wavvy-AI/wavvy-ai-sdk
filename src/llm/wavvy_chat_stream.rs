use std::task::Poll;

use crate::prompt_template::chat_template::Model;

use super::token_output::TokenOutput;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2;
use futures::Stream;
use thiserror::Error;
use tokenizers::{Encoding, Tokenizer};

#[derive(Error, Debug)]
pub enum WavvyError {
    #[error("Config error, {0}")]
    ConfigError(String),
    #[error("Tokenizer error, {0}")]
    TokenizerError(String),
    #[error("Prompt error, {0}")]
    PromptError(String),
}

pub struct WavvyChatStream {
    model: Model,
    base_model: Qwen2,
    device: Device,
    tos: TokenOutput,
    all_tokens: Vec<u32>,
    eos_token: u32,
    index: usize,
    next_token: u32,
    tokens: Encoding,
    token_ids: Vec<u32>,
    logits_processor: LogitsProcessor,
    is_prompt_initialized: bool,
    pub args: WavvyArgs,
}

#[derive(Debug)]
pub struct ChatResponse {
    pub content: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone, Debug)]
pub struct WavvyArgs {
    pub sample_len: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub seed: u64,
    pub split_prompt: bool,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for WavvyArgs {
    fn default() -> Self {
        Self {
            sample_len: 1000,
            temperature: 0.8,
            top_p: None,
            top_k: None,
            seed: 299792458,
            split_prompt: true,
            repeat_penalty: 1.1,
            repeat_last_n: 65,
        }
    }
}

impl WavvyChatStream {
    pub fn new(
        model: Model,
        base_model: Qwen2,
        tokenizer: Tokenizer,
        device: &Device,
        args: Option<WavvyArgs>,
    ) -> Self {
        let default_args = WavvyArgs::default();
        Self {
            model,
            base_model,
            device: device.clone(),
            tos: TokenOutput::new(tokenizer),
            all_tokens: vec![],
            eos_token: 0,
            index: 0,
            next_token: 0,
            tokens: Encoding::default(),
            token_ids: vec![],
            logits_processor: LogitsProcessor::from_sampling(default_args.seed, Sampling::ArgMax),
            is_prompt_initialized: false,
            args: args.clone().unwrap_or(default_args),
        }
    }

    fn init_logits_processor(&self) -> LogitsProcessor {
        let temperature = self.args.temperature;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (self.args.top_k, self.args.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(self.args.seed, sampling)
    }

    fn prompt_next_token(&mut self) -> Result<u32, WavvyError> {
        let next_token = if !self.args.split_prompt {
            let input = Tensor::new(self.token_ids.clone(), &self.device)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?;
            let logits = self.base_model.forward(&input, 0).unwrap();
            let logits = logits
                .squeeze(0)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?;
            self.logits_processor
                .sample(&logits)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?
        } else {
            let mut next_token = 0;
            for (pos, token) in self.token_ids.iter().enumerate() {
                let input = Tensor::new(&[*token], &self.device)
                    .map_err(|e| WavvyError::PromptError(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| WavvyError::PromptError(e.to_string()))?;
                let logits = self
                    .base_model
                    .forward(&input, pos)
                    .map_err(|e| WavvyError::PromptError(e.to_string()))?;
                let logits = logits
                    .squeeze(0)
                    .map_err(|e| WavvyError::PromptError(e.to_string()))?;
                next_token = self
                    .logits_processor
                    .sample(&logits)
                    .map_err(|e| WavvyError::PromptError(e.to_string()))?;
            }
            next_token
        };
        Ok(next_token)
    }

    pub fn process_logits(
        &mut self,
        next_token: u32,
        index: usize,
        all_tokens: Vec<u32>,
    ) -> Result<Tensor, WavvyError> {
        let input = Tensor::new(&[next_token], &self.device)
            .map_err(|e| WavvyError::PromptError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| WavvyError::PromptError(e.to_string()))?;

        let logits = self
            .base_model
            .forward(&input, self.token_ids.len() + index)
            .map_err(|e| WavvyError::PromptError(e.to_string()))?;

        let logits = logits
            .squeeze(0)
            .map_err(|e| WavvyError::PromptError(e.to_string()))?;

        let logits = if self.args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(self.args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.args.repeat_penalty,
                &all_tokens[start_at..],
            )
            .map_err(|e| WavvyError::PromptError(e.to_string()))?
        };
        Ok(logits)
    }

    pub fn invoke(mut self, prompt_str: String) -> Result<Self, WavvyError> {
        if self.model == Model::R1 {
            self.eos_token = *self
                .tos
                .tokenizer()
                .get_vocab(true)
                .get("<｜end▁of▁sentence｜>")
                .unwrap();
        } else if self.model == Model::W {
            self.eos_token = *self
                .tos
                .tokenizer()
                .get_vocab(true)
                .get("<|im_end|>")
                .unwrap();
        }

        self.tokens = self
            .tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(|e| WavvyError::TokenizerError(e.to_string()))?;

        self.token_ids = self.tokens.get_ids().to_vec();

        self.logits_processor = self.init_logits_processor();
        self.next_token = self.prompt_next_token()?;

        Ok(self)
    }
}

impl Stream for WavvyChatStream {
    type Item = Result<ChatResponse, WavvyError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();

        if this.index == this.args.sample_len.saturating_sub(1) {
            return Poll::Ready(None);
        }

        if this.next_token == this.eos_token {
            return Poll::Ready(None);
        }

        if !this.is_prompt_initialized {
            if let Some(text) = this
                .tos
                .next_token(this.next_token)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?
            {
                this.is_prompt_initialized = true;
                this.all_tokens.push(this.next_token);
                let prompt_tokens = this.token_ids.len();
                let completion_tokens = this.tos.total_tokens();
                return Poll::Ready(Some(Ok(ChatResponse {
                    content: text,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                })));
            }
        }

        let logits = this.process_logits(this.next_token, this.index, this.all_tokens.clone())?;

        this.next_token = this
            .logits_processor
            .sample(&logits)
            .map_err(|e| WavvyError::PromptError(e.to_string()))?;

        this.all_tokens.push(this.next_token);
        let text = this
            .tos
            .next_token(this.next_token)
            .map_err(|e| WavvyError::PromptError(e.to_string()))?;

        this.index += 1;

        let prompt_tokens = this.token_ids.len();
        let completion_tokens = this.tos.total_tokens();
        return Poll::Ready(Some(Ok(ChatResponse {
            content: text.unwrap_or_default(),
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        })));
    }
}
