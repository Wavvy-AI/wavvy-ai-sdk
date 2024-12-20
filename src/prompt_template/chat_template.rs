use mustache::Data;

use super::message::Message;

#[derive(Debug)]
pub struct ChatTemplate {
    pub messages: Vec<Message>,
}

impl ChatTemplate {
    pub fn new(messages: Vec<Message>) -> Self {
        return Self { messages };
    }

    pub fn format(&self) -> String {
        let mut msg: String = String::new();
        for message in &self.messages {
            msg.push_str(&message.encode());
            msg.push_str("\n");
        }
        msg.push_str("<|im_start|>assistant\n");
        msg
    }

    pub fn format_with_params(&self, data: &Data) -> String {
        let text_msg = self.format();

        let mut bytes = vec![];

        let template = mustache::compile_str(&text_msg).unwrap();
        template.render_data(&mut bytes, data).unwrap();

        String::from_utf8(bytes).unwrap()
    }
}
