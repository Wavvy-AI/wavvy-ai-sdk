use std::fmt;

#[derive(Debug)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
        }
    }
}

#[derive(Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn encode(&self) -> String {
        format!(
            "<|im_start|>{}\n{}<|im_end|>",
            self.role.to_string(),
            self.content
        )
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.role, self.content)
    }
}
