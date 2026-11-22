"""The instruction prompt. The SAME prompt is used for the base model and the fine-tuned model, so the
comparison measures only the effect of fine-tuning."""
import json

from config import LABELS

SYSTEM_PROMPT = f"""You triage customer messages sent to an online shop in Pakistan on WhatsApp.
Messages can be in English, Roman Urdu (Urdu written in English letters) or a mix of both.

Return ONLY a JSON object with exactly these keys:
"category": one of {", ".join(LABELS["category"])}
"urgency": one of {", ".join(LABELS["urgency"])}
"sentiment": one of {", ".join(LABELS["sentiment"])}

Category meanings:
- damaged_item: the item arrived broken, defective, leaking or not working
- wrong_item: wrong product, size or colour was sent, or an item is missing from the order
- delivery_issue: late, delayed, lost or not delivered order, courier or tracking problems
- refund_request: asking for a refund, or about the status or amount of a refund
- return_exchange: wants to return or exchange an item, or asks about the return policy
- payment_issue: payment failed, charged twice, money deducted, cash on delivery or payment method questions
- cancellation: wants to cancel an order
- general_inquiry: questions about products, price, stock, size or material
- other: thanks, praise, greetings or anything unrelated

Urgency: high = angry or threatening, money lost, item broken or missing, or an explicit deadline;
medium = a real problem stated calmly; low = simple questions, thanks or praise.

Sentiment: the customer's tone (negative, neutral or positive)."""


def messages_for(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]


def target_json(row: dict) -> str:
    """The answer the fine-tuned model learns to produce."""
    return json.dumps({task: row[task] for task in ("category", "urgency", "sentiment")})
