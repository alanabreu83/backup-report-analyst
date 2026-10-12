from dotenv import load_dotenv
import anthropic

load_dotenv()  # reads ANTHROPIC_API_KEY from .env
client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=200,
    messages=[{"role": "user", "content": "In one sentence, why do backups silently fail?"}],
)
print(message.content[0].text)