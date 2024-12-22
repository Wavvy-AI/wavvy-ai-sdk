use super::wavvy_chat_stream::{ChatResponse, WavvyArgs, WavvyChatStream, WavvyError};
use candle_core::Device;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2;
use futures::StreamExt;
use tokenizers::Tokenizer;

pub struct WavvyChat {
    model: Qwen2,
    device: Device,
    tokenizer: Tokenizer,
    pub args: WavvyArgs,
}

impl WavvyChat {
    pub fn new(
        model: Qwen2,
        tokenizer: Tokenizer,
        device: &Device,
        args: Option<WavvyArgs>,
    ) -> Self {
        Self {
            model,
            device: device.clone(),
            tokenizer,
            args: args.clone().unwrap_or_default(),
        }
    }

    async fn process_invoke(self, prompt_str: String) -> Result<ChatResponse, WavvyError> {
        let wavvy = WavvyChatStream::new(self.model, self.tokenizer, &self.device, Some(self.args));
        let mut wavvy_response = wavvy.invoke(prompt_str).unwrap();
        let mut resp = ChatResponse {
            content: String::default(),
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        };

        while let Some(item) = wavvy_response.next().await {
            match item {
                Ok(response) => {
                    resp.content.push_str(response.content.as_str());
                    resp.prompt_tokens = response.prompt_tokens;
                    resp.completion_tokens = response.completion_tokens;
                    resp.total_tokens = response.total_tokens;
                }
                Err(e) => {
                    println!("Error: {}", e);
                }
            }
        }
        Ok(resp)
    }

    pub fn invoke(self, prompt_str: String) -> Result<ChatResponse, WavvyError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| WavvyError::ConfigError(e.to_string()))?;
        let response = runtime.block_on(self.process_invoke(prompt_str))?;
        Ok(response)
    }

    pub fn stream_invoke(self, prompt_str: String) -> Result<WavvyChatStream, WavvyError> {
        let wavvy = WavvyChatStream::new(self.model, self.tokenizer, &self.device, Some(self.args));
        let wavvy_stream = wavvy.invoke(prompt_str)?;
        Ok(wavvy_stream)
    }
}
