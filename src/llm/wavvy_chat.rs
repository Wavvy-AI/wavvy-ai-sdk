use super::token_output_stream::TokenOutputStream;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Error, Debug)]
pub enum WavvyError {
    #[error("Tokenizer error, {0}")]
    TokenizerError(String),
    #[error("Prompt error, {0}")]
    PromptError(String),
}

pub struct WavvyChat {
    model: ModelWeights,
    device: Device,
    all_tokens: Vec<u32>,
    tos: TokenOutputStream,
    eos_token: u32,
    to_sample: usize,
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

impl WavvyChat {
    pub fn new(
        model: ModelWeights,
        tokenizer: Tokenizer,
        device: &Device,
        args: Option<WavvyArgs>,
    ) -> Self {
        Self {
            model,
            device: device.clone(),
            all_tokens: Vec::default(),
            tos: TokenOutputStream::new(tokenizer),
            eos_token: 0,
            to_sample: 0,
            args: args.clone().unwrap_or(WavvyArgs::default()),
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

    pub fn invoke(&mut self, prompt_str: String) -> Result<ChatResponse, WavvyError> {
        let tokens = self
            .tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(|e| WavvyError::TokenizerError(e.to_string()))?;

        let tokens = tokens.get_ids();

        let mut logits_processor = self.init_logits_processor();

        let mut next_token = if !self.args.split_prompt {
            let input = Tensor::new(tokens, &self.device)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?;
            let logits = self.model.forward(&input, 0).unwrap();
            let logits = logits
                .squeeze(0)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?;
            logits_processor
                .sample(&logits)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?
        } else {
            let mut next_token = 0;
            for (pos, token) in tokens.iter().enumerate() {
                let input = Tensor::new(&[*token], &self.device)
                    .map_err(|e| WavvyError::PromptError(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| WavvyError::PromptError(e.to_string()))?;
                let logits = self
                    .model
                    .forward(&input, pos)
                    .map_err(|e| WavvyError::PromptError(e.to_string()))?;
                let logits = logits
                    .squeeze(0)
                    .map_err(|e| WavvyError::PromptError(e.to_string()))?;
                next_token = logits_processor
                    .sample(&logits)
                    .map_err(|e| WavvyError::PromptError(e.to_string()))?;
            }
            next_token
        };

        self.tos
            .next_token(next_token)
            .map_err(|e| WavvyError::PromptError(e.to_string()))?;

        self.all_tokens.push(next_token);

        self.eos_token = *self
            .tos
            .tokenizer()
            .get_vocab(true)
            .get("<|im_end|>")
            .unwrap();

        self.to_sample = self.args.sample_len.saturating_sub(1);

        let mut completion_tokens = 0;
        for index in 0..self.to_sample {
            let input = Tensor::new(&[next_token], &self.device)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?;

            let logits = self
                .model
                .forward(&input, tokens.len() + index)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?;

            let logits = logits
                .squeeze(0)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?;

            let logits = if self.args.repeat_penalty == 1. {
                logits
            } else {
                let start_at = self
                    .all_tokens
                    .len()
                    .saturating_sub(self.args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.args.repeat_penalty,
                    &self.all_tokens[start_at..],
                )
                .map_err(|e| WavvyError::PromptError(e.to_string()))?
            };

            next_token = logits_processor
                .sample(&logits)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?;
            self.all_tokens.push(next_token);

            self.tos
                .next_token(next_token)
                .map_err(|e| WavvyError::PromptError(e.to_string()))?;

            completion_tokens += 1;

            if next_token == self.eos_token {
                break;
            };
        }

        let content = self
            .tos
            .decode_all()
            .map_err(|e| WavvyError::PromptError(e.to_string()))?;

        Ok(ChatResponse {
            content,
            prompt_tokens: tokens.len(),
            completion_tokens,
            total_tokens: tokens.len() + completion_tokens,
        })
    }
}
