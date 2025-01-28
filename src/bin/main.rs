use clap::Parser;

use candle_core::Device;
use futures::StreamExt;
use wavvy_ai_sdk::{
    llm::{model_builder::ModelBuilder, wavvy_chat::WavvyChat, wavvy_chat_stream::WavvyArgs},
    prompt_template::{
        chat_template::{ChatTemplate, Model},
        message::Message,
        role::Role,
    },
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(
        long,
        help = "The model name to use 'r1' or 'w' default is 'w'",
        default_value_t = String::from("w"),
    )]
    pub model_name: String,

    #[arg(long)]
    pub model_path: Option<String>,

    #[arg(long)]
    pub tokenizer_path: Option<String>,

    #[arg(long)]
    pub prompt: Option<String>,

    #[arg(short = 'n', long, default_value_t = 1000)]
    pub sample_len: usize,

    #[arg(long, default_value_t = 0.8)]
    pub temperature: f64,

    #[arg(long)]
    pub top_p: Option<f64>,

    #[arg(long)]
    pub top_k: Option<usize>,

    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,

    #[arg(long)]
    pub split_prompt: bool,

    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,

    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let device = Device::new_metal(0).unwrap();

    let question = args.prompt.unwrap();

    let model_name = if args.model_name == "r1" {
        Model::R1
    } else {
        Model::W
    };

    let messages = vec![Message::new(Role::User, question.clone())];

    let message_template = ChatTemplate::new(model_name, messages);

    // Path examples:
    // model-path: ./model/Qwen2.5-3B-Instruct/qwen2.5-3b-instruct-q4_0.gguf
    // tokenizer-path: ./model/Qwen2.5-3B-Instruct/tokenizer.json
    let model_builder = ModelBuilder::new(
        args.model_path.unwrap().as_str(),
        args.tokenizer_path.unwrap().as_str(),
        &device,
    );

    let tokenizer = model_builder.load_tokenizer();
    let model = model_builder.load_model();
    println!("Model and tokenizer loaded");

    let wavvy_args = Some(WavvyArgs {
        sample_len: args.sample_len,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        seed: args.seed,
        split_prompt: args.split_prompt,
        repeat_penalty: args.repeat_penalty,
        repeat_last_n: args.repeat_last_n,
    });

    let model_name = if args.model_name == "r1" {
        Model::R1
    } else {
        Model::W
    };

    let wavvy = WavvyChat::new(model_name, model, tokenizer, &device, wavvy_args);
    let mut response = wavvy.stream_invoke(message_template.format()).unwrap();

    println!("Question: {question}");
    print!("Answer: ");
    while let Some(item) = response.next().await {
        match item {
            Ok(response) => {
                print!("{}", response.content);
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }
}
