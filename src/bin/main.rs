use candle_core::Device;
// use futures::StreamExt;
use wavvy_engine::{
    llm::{model_builder::ModelBuilder, wavvy_chat::WavvyChat},
    prompt_template::{
        chat_template::ChatTemplate,
        message::{Message, Role},
    },
};

// #[tokio::main]
// async fn run_stream() {
//     let assistant = Message {
//         role: Role::System,
//         content: String::from("You are helpful assistant!"),
//     };

//     let question = "What is 1+1";

//     let messages = vec![
//         assistant,
//         Message {
//             role: Role::User,
//             content: String::from(question),
//         },
//     ];

//     let message_template = ChatTemplate::new(messages);

//     let device = Device::new_metal(0).unwrap();
//     let model_builder = ModelBuilder::new(
//         "./model/Qwen2.5-3B-Instruct/qwen2.5-3b-instruct-q4_0.gguf",
//         "./model/Qwen2.5-3B-Instruct/tokenizer.json",
//         &device,
//     );

//     let tokenizer = model_builder.load_tokenizer();
//     let model = model_builder.load_model();
//     println!("Model and tokenizer loaded");

//     println!("Question: {question}");
//     print!("Answer: ");
//     let wavvy = WavvyChat::new(model, tokenizer, &device, None);
//     let mut response = wavvy.stream_invoke(message_template.format()).unwrap();
//     while let Some(item) = response.next().await {
//         match item {
//             Ok(response) => {
//                 print!("{}", response.content);
//             }
//             Err(e) => {
//                 println!("Error: {}", e);
//             }
//         }
//     }
// }

fn main() {
    let assistant = Message {
        role: Role::System,
        content: String::from("You are helpful assistant!"),
    };

    let question = "What is 1+1";

    let messages = vec![
        assistant,
        Message {
            role: Role::User,
            content: String::from(question),
        },
    ];

    let message_template = ChatTemplate::new(messages);

    let device = Device::new_metal(0).unwrap();
    let model_builder = ModelBuilder::new(
        "./model/Qwen2.5-3B-Instruct/qwen2.5-3b-instruct-q4_0.gguf",
        "./model/Qwen2.5-3B-Instruct/tokenizer.json",
        &device,
    );

    println!("Wavvy initializing...");

    let tokenizer = model_builder.load_tokenizer();
    let model = model_builder.load_model();
    println!("Model and tokenizer loaded");

    let wavvy = WavvyChat::new(model, tokenizer, &device, None);
    println!("Question: {question}");
    let response = wavvy.invoke(message_template.format()).unwrap();
    println!("Answer: {}", response.content);
    println!("Prompt Tokens: {}", response.prompt_tokens);
    println!("Completion Tokens: {}", response.completion_tokens);
    println!("Total Tokens: {}", response.total_tokens);
}
