use candle_core::{quantized::gguf_file, Device};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct ModelBuilder {
    pub model_path: String,
    pub tokenizer_path: String,
    pub device: Device,
}

impl ModelBuilder {
    pub fn new(model_path: &str, tokenizer_path: &str, device: &Device) -> Self {
        Self {
            model_path: model_path.to_string(),
            tokenizer_path: tokenizer_path.to_string(),
            device: device.clone(),
        }
    }

    pub fn load_tokenizer(&self) -> Tokenizer {
        Tokenizer::from_file(std::path::PathBuf::from(&self.tokenizer_path)).unwrap()
    }

    pub fn load_model(&self) -> ModelWeights {
        let model_path = std::path::PathBuf::from(&self.model_path);
        let mut file = std::fs::File::open(&model_path).unwrap();
        let model: gguf_file::Content = gguf_file::Content::read(&mut file)
            .map_err(|e| e.with_path(model_path))
            .unwrap();
        ModelWeights::from_gguf(model, &mut file, &self.device).unwrap()
    }
}
