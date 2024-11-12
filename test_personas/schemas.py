from pydantic import BaseModel, Field

class InputSchema(BaseModel):
    question: str = Field(..., title="Chat question")
    num_personas: int = Field(..., title="Number of personas")