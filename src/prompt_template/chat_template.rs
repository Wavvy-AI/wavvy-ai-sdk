use mustache::Data;

use super::message::Message;

#[derive(Debug, PartialEq)]
pub enum Model {
    W,
    R1,
}

#[derive(Debug)]
pub struct ChatTemplate {
    pub messages: Vec<Message>,
    pub model: Model,
}

impl ChatTemplate {
    pub fn new(model: Model, messages: Vec<Message>) -> Self {
        return Self { messages, model };
    }

    pub fn format(&self) -> String {
        let mut msg: String = String::new();
        for message in &self.messages {
            if self.model == Model::W {
                let p_msg = format!(
                    "<|im_start|>{}\n{}<|im_end|>",
                    message.to_string(),
                    message.content
                );
                msg.push_str(p_msg.as_str());
            } else if self.model == Model::R1 {
                let role = message.role.to_string();
                let mut c = role.as_str().chars();
                let cap_role = match c.next() {
                    None => String::new(),
                    Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                };
                let p_msg = format!("<｜{}｜>{}", cap_role, message.content);
                msg.push_str(p_msg.as_str());
            }
            msg.push_str("\n");
        }
        if self.model == Model::W {
            msg.push_str("<|im_start|>assistant\n");
        } else if self.model == Model::R1 {
            msg.push_str("<｜Assistant｜>");
        }
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
