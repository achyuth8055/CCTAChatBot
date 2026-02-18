import os
import re
import sys
import io
import contextlib
import uuid
import traceback
from datetime import datetime

from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
import json
import time
import random
import logging

from models import Document, DocumentStatus, SessionLocal, init_db

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("posthog").setLevel(logging.ERROR)

app = Flask(__name__)

CHROMA_PATH = "chroma_db"
LLM_MODEL = os.getenv("DEEPSEEK_MODEL") if os.getenv("DEEPSEEK_API_KEY") else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
WEBSITE_URL = os.getenv("CCTA_WEBSITE", "https://www.cookcountyllc.com")
RETRIEVAL_SCORE_THRESHOLD = 0.20
MIN_CONTEXT_LENGTH = 50

ALLOWED_SERVICES = {
    "tax_law": ["tax law", "IRS representation", "tax litigation", "tax defense", "tax planning"],
    "estate_planning": ["estate planning", "wills", "trusts", "asset protection", "probate"],
    "business_law": ["business law", "corporate law", "entity formation", "contracts", "mergers", "acquisitions"],
    "real_estate": ["real estate", "real estate transactions", "property law"],
    "property_tax": ["property tax appeals", "property tax"],
    "personal_injury": ["personal injury", "car accidents", "accident cases", "injury settlements"]
}

ALL_ALLOWED_SERVICES = [s for services in ALLOWED_SERVICES.values() for s in services]

_stderr_suppressor = io.StringIO()
with contextlib.redirect_stderr(_stderr_suppressor):
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(name="cookcounty_tax_faqs")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com" if os.getenv("DEEPSEEK_API_KEY") else "https://api.openai.com/v1"
)


class RequestContext:
    def __init__(self):
        self.request_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
    
    def elapsed(self):
        return round(time.time() - self.start_time, 3)


def detect_intents(user_query: str) -> list:
    query_lower = user_query.lower()
    intents = []
    
    appointment_keywords = [
        'book', 'schedule', 'appointment', 'meeting', 'reserve', 'slot',
        'book online', 'schedule online', 'choose a time', 'pick a time',
        'available time', 'availability', 'book a consultation',
        'make an appointment', 'set up appointment', 'arrange meeting'
    ]
    if any(keyword in query_lower for keyword in appointment_keywords):
        intents.append('appointment_booking')
        return intents
    
    if any(phrase in query_lower for phrase in ['i need', 'help me', 'need help', 'struggling', 'i had', 'i got']):
        intents.append('empathy')
    
    if any(phrase in query_lower for phrase in ['do you handle', 'do you do', 'do you guys', 'can you help with']):
        if any(word in query_lower for word in ['accident', 'injury', 'personal injury', 'car accident', 'crash']):
            intents.append('accident_yesno')
        elif any(word in query_lower for word in ['service', 'case', 'matter']):
            intents.append('services_yesno')
    elif any(word in query_lower for word in ['accident', 'injury', 'hurt', 'crash']):
        intents.append('accident_info')
    
    if any(phrase in query_lower for phrase in ['what service', 'what do you', 'what legal', 'services do you', 'what areas', 'services handled', 'what you handle', 'services you handle', 'what all services', 'services you offer', 'what do you offer', 'what do you provide']):
        intents.append('services_list')
    
    if any(word in query_lower for word in ['fee', 'cost', 'price', 'pricing', 'charge', 'how much', 'pay']):
        intents.append('pricing')
    
    contact_phrases = [
        'how do i contact', 'how to contact', 'reach you', 'get in touch',
        'contact info', 'contact information', 'contact details',
        'office details', 'office info', 'office information',
        'your details', 'your info', 'your information',
        'firm details', 'firm info', 'company info',
        'ways to contact', 'how can i reach', 'how to reach'
    ]
    
    contact_context_words = ['office', 'contact', 'firm', 'company']
    detail_words = ['details', 'info', 'information']
    
    is_contact_query = (
        any(phrase in query_lower for phrase in contact_phrases) or
        (any(ctx in query_lower for ctx in contact_context_words) and 
         any(det in query_lower for det in detail_words))
    )
    
    if is_contact_query:
        intents.append('contact_methods')
    elif 'phone' in query_lower or 'number' in query_lower or 'call' in query_lower:
        intents.append('phone')
    elif 'email' in query_lower or 'e-mail' in query_lower:
        intents.append('email')
    elif 'address' in query_lower or 'location' in query_lower or 'where are you' in query_lower or 'office' in query_lower or 'where can i' in query_lower or 'come see you' in query_lower or 'visit you' in query_lower or 'located' in query_lower or 'find you' in query_lower or 'directions' in query_lower:
        intents.append('address')
    
    if any(phrase in query_lower for phrase in ['open today', 'are you open', 'business hours', 'what time', 'when open']):
        intents.append('hours')
    
    if not intents:
        intents.append('general')
    
    return intents


def classify_contact_intent(user_query: str, intents: list) -> str:
    query_lower = user_query.lower()
    
    consultation_keywords = [
        'contact', 'phone', 'call', 'email', 'reach', 'get in touch',
        'speak to', 'talk to', 'consult', 'hire', 'retain',
        'schedule', 'appointment', 'book', 'meeting',
        'next step', 'what now', 'how do i proceed', 'get started',
        'sign up', 'work with you',
        'office details', 'office info', 'contact details', 'contact info',
        'your details', 'your info', 'firm details', 'company info'
    ]
    if any(kw in query_lower for kw in consultation_keywords):
        return 'consultation_ready'
    
    if 'office' in query_lower and any(w in query_lower for w in ['details', 'info', 'information', 'address', 'location']):
        return 'consultation_ready'
    
    contact_intents = ['phone', 'email', 'address', 'contact_methods', 'appointment_booking']
    if any(intent in intents for intent in contact_intents):
        return 'consultation_ready'
    
    service_inquiry_keywords = [
        'can you help', 'do you handle', 'do you take', 'can you take',
        'eligible', 'qualify', 'my case', 'my situation',
        'fee', 'cost', 'price', 'how much', 'charge', 'payment',
        'free consultation', 'consultation fee',
        'i need a lawyer', 'i need an attorney', 'i need help with',
        'represent me', 'help me with'
    ]
    if any(kw in query_lower for kw in service_inquiry_keywords):
        return 'service_inquiry'
    
    if any(intent in intents for intent in ['pricing', 'empathy', 'accident_yesno', 'services_yesno']):
        return 'service_inquiry'
    
    exploratory_keywords = [
        'what service', 'what do you', 'what areas', 'practice areas',
        'what kind of', 'types of cases', 'specializ'
    ]
    if any(kw in query_lower for kw in exploratory_keywords):
        return 'exploratory'
    
    if 'services_list' in intents:
        return 'exploratory'
    
    return 'informational'


def should_include_contact(contact_category: str) -> bool:
    return contact_category in ['service_inquiry', 'consultation_ready']


def strip_contact_info(response: str) -> str:
    patterns = [
        r'\n*📞[^\n]*',
        r'\n*📧[^\n]*',
        r'\n*📍[^\n]*',
        r'\n*🌐[^\n]*',
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        r'\|\s*📧[^\n]*',
        r'contact us at[^.]*\.',
        r'call us at[^.]*\.',
        r'reach out[^.]*\.',
        r'for more information,?\s*(please)?\s*(call|contact|reach)[^.]*\.',
    ]
    
    result = response
    for pattern in patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = re.sub(r'\s+$', '', result, flags=re.MULTILINE)
    result = result.strip()
    
    return result


def build_retrieval_query(intent: str, user_query: str) -> str:
    """Build retrieval query. Always uses the user's original query,
    optionally augmented with helpful keywords for better retrieval."""
    augment_map = {
        'services_list': 'services offered available',
        'services_yesno': 'services handle provide',
        'accident_yesno': 'handle cases services',
        'accident_info': 'handle cases services',
        'pricing': 'fees pricing cost charge payment',
        'phone': 'phone number contact telephone',
        'email': 'email address contact',
        'address': 'office address location',
        'hours': 'business hours open schedule',
    }
    extra = augment_map.get(intent, '')
    if extra:
        return f"{user_query} {extra}"
    return user_query


def retrieve_with_quality_check(intent: str, user_query: str, ctx: RequestContext) -> dict:
    try:
        retrieval_query = build_retrieval_query(intent, user_query)
        
        logger.info(f"[{ctx.request_id}] Intent: {intent}, Query: '{retrieval_query[:50]}...'")
        
        with contextlib.redirect_stderr(_stderr_suppressor):
            results = collection.query(
                query_texts=[retrieval_query],
                n_results=5,
                include=["documents", "distances"]
            )
        
        documents = results.get("documents", [[]])[0] if results.get("documents") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        
        if not documents:
            logger.warning(f"[{ctx.request_id}] No documents retrieved")
            return {"success": False, "context": "", "metadata": {"reason": "no_results"}}
        
        scores = [max(0, 1 - (d / 2)) for d in distances]
        
        good_docs = []
        good_scores = []
        for doc, score in zip(documents, scores):
            if score >= RETRIEVAL_SCORE_THRESHOLD and len(doc.strip()) >= MIN_CONTEXT_LENGTH:
                good_docs.append(doc)
                good_scores.append(score)
        
        logger.info(f"[{ctx.request_id}] Retrieved: {len(documents)} docs, Good: {len(good_docs)}, " +
                   f"Scores: {[round(s, 2) for s in good_scores[:3]]}")
        
        if not good_docs:
            logger.warning(f"[{ctx.request_id}] All results below quality threshold")
            return {"success": False, "context": "", "metadata": {"reason": "low_quality", "scores": scores}}
        
        context = "\n\n".join(good_docs[:3])
        
        return {
            "success": True,
            "context": context,
            "metadata": {
                "num_docs": len(good_docs),
                "top_scores": good_scores[:3],
                "query": retrieval_query
            }
        }
    
    except Exception as e:
        logger.error(f"[{ctx.request_id}] Retrieval error: {str(e)}")
        return {"success": False, "context": "", "metadata": {"reason": "error", "error": str(e)}}


FIRM_CONTACT = {
    "name": "Cook County Tax Appeals, LLC",
    "address": "6600 College Drive, Suite 207, Palos Heights, IL 60463",
    "email": "info@cookcountytaxappeal.com",
    "phone": "708-888-8880",
    "website": "https://www.cookcountytaxappeal.com",
}

# share teh below prompt witht he LLM so that it can reframe the responce more naturally and remove any robotic phrases, and also decide whether to include contact info based on the intent and conversation flow. The LLM should strictly follow the instructions in the prompt to ensure accurate, helpful, and human-like responses while adhering to the anti-hallucination rules and formatting guidelines.
def build_rag_system_prompt(context: str, intent: str, include_contact: bool = False) -> str:
    contact_instruction = """

CONTACT INFO RULES:
- Include contact info ONLY if the user explicitly asks for it
- Place contact info at the END of your response, never in the middle
- Do NOT proactively offer contact details unless asked""" if not include_contact else """

CONTACT INFO (include at end when contextually appropriate — use the VERIFIED details above)"""

    today = datetime.now().strftime("%B %d, %Y")
    fc = FIRM_CONTACT

    return f"""You are a warm, knowledgeable assistant at Cook County Tax Appeals, LLC (CCTA) — a real person on the team helping visitors.

TODAY'S DATE: {today}

VERIFIED CONTACT INFORMATION (use ONLY these — never invent or guess):
📍 Address: {fc['address']}
📧 Email: {fc['email']}
📞 Phone: {fc['phone']}
🌐 Website: {fc['website']}

PERSONALITY:
- Friendly, approachable, and confident — like a helpful colleague
- Use natural conversational language, not corporate jargon
- Vary your sentence structure — mix short and medium sentences
- It's okay to use casual transitions like "Actually," "By the way," "That said,"
- Show genuine interest in helping

OUTPUT FORMAT:
1. Plain text with simple formatting only
2. Use • for bullet lists; always include a brief intro line before a list
3. Max one blank line between sections
4. Short paragraphs (1-3 sentences)

DATE & DEADLINE RULES (CRITICAL — follow strictly):
1. Today's date is {today}. Use it to evaluate ALL deadlines.
2. If a deadline date is BEFORE today → that township/item is CLOSED. Do NOT list it as open.
3. If a deadline date is AFTER today → that township/item is OPEN. Include the deadline.
4. If something says "Open until [date]" and that date has passed, it is CLOSED — say so clearly.
5. When the user asks what is "open", ONLY list items whose deadline is in the future.
6. If everything is closed, say so honestly and suggest the user check back or contact the office for updated dates.
7. For "pre-file" items with no specific deadline, mention them as upcoming/pre-file — not as open.

ANTI-HALLUCINATION RULES (CRITICAL):
1. For addresses, phone numbers, emails, and website — use ONLY the VERIFIED CONTACT INFORMATION above
2. NEVER invent, guess, or fabricate any factual details like addresses, names, numbers, or dates
3. If you don't know something, say so — don't make it up
4. If the provided information doesn't answer the question, say you're not sure and suggest contacting the office

CONVERSATION CONTINUITY:
- You are in a multi-turn conversation. Prior messages are included for context.
- Reference what was just discussed when relevant (e.g., "As I mentioned…", "Building on that…", "Since you asked about X…")
- If the user says "that", "it", "this", "more", etc., connect it to the most recent topic
- Keep the conversation flowing naturally — don't repeat yourself or re-introduce what you already said
- Vary your openings — don't start every response the same way

CONTENT RULES:
1. Answer ONLY what the user asks — stay focused
2. Use ONLY the information provided below — do not invent facts
3. NEVER mention documents, context, retrieval, databases, or any internal system
4. NEVER say "based on", "according to", "the information shows", or similar
5. Speak as "we" / "our" — you represent the firm ("We handle…", "Our team…")
6. Do NOT append contact info unless the user asks for it{contact_instruction}

RESPONSE LENGTH:
- Simple questions → 1-2 sentences
- Service inquiries → brief intro + concise bullet list
- Yes/No → answer first, then a short explanation

USER INTENT: {intent}

INFORMATION:
{context}

Respond naturally and helpfully. Answer the question directly."""


def format_response_by_intent(intent: str, rag_response: str, user_query: str) -> str:
    response = rag_response.strip()
    
    robotic_phrases = [
        r'based on the (context|information|documents?),?\s*',
        r'according to the (documents?|information|context),?\s*',
        r'the (context|information|documents?) (shows?|indicates?|states?|mentions?|suggests?) that\s*',
        r'from the (documents?|information|context),?\s*',
        r'in the (uploaded |provided )?documents?,?\s*',
        r'the (provided |uploaded )?information (shows?|indicates?) that\s*',
        r'as (mentioned|stated|indicated) in the (context|documents?),?\s*',
        r'per the (context|information|documents?),?\s*',
        r'the (knowledge base|files?|resources?) (show|indicate|mention)\s*',
        r'from what I (can see|understand|gather) in the (context|documents?),?\s*',
        r'looking at the (context|information|documents?),?\s*',
        r"here (is|are) the services?\s*",
        r"here's (what|a list of)\s*"
    ]
    
    for phrase in robotic_phrases:
        response = re.sub(phrase, '', response, flags=re.IGNORECASE)
    
    response = re.sub(r'\n{3,}', '\n\n', response)
    response = re.sub(r'^\n+', '', response)
    response = re.sub(r'\n+$', '', response)
    response = re.sub(r'^[-*]\s+', '• ', response, flags=re.MULTILINE)
    response = re.sub(r'[ \t]+', ' ', response)
    response = re.sub(r'—+', '', response)
    response = response.strip()
    if response and response[0].islower():
        response = response[0].upper() + response[1:]
    
    response_lower = response.lower()
    if 'we handle' in response_lower and 'do not handle' in response_lower:
        lines = response.split('.')
        response = lines[0] + '.' if lines else response
    
    if intent in ['services_list', 'services_yesno']:
        if '•' in response and not any(intro in response.lower() for intro in ['we offer', 'we provide', 'our services', 'services include']):
            lines = response.split('\n')
            if lines and lines[0].startswith('•'):
                response = "We provide the following legal services:\n" + response
    
    if intent == 'empathy' and not any(response.lower().startswith(w) for w in ["i'm sorry", "that's", "sorry to hear"]):
        if 'accident' in user_query.lower() or 'hurt' in user_query.lower() or 'injury' in user_query.lower():
            response = "I'm sorry to hear that. " + response
    
    return response


def _build_chat_messages(system_prompt: str, user_query: str, history: list) -> list:
    """Build the messages array for the LLM, including conversation history."""
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    
    messages.append({"role": "user", "content": user_query})
    return messages


def _resolve_vague_query(user_query: str, history: list) -> str:
    """If the user's message is vague (e.g. 'tell me more', 'what about that'),
    rewrite it using recent conversation history for better retrieval."""
    q = user_query.lower().strip()
    
    vague_patterns = [
        'tell me more', 'more about that', 'what about that', 'what about it',
        'explain that', 'can you elaborate', 'elaborate', 'go on',
        'more details', 'more info', 'what else', 'anything else',
        'and', 'also', 'what about this', 'expand on that',
        'yes tell me', 'yes please', 'yes', 'continue',
    ]
    
    is_vague = (
        len(q.split()) <= 5 and any(v in q for v in vague_patterns)
    ) or q in ('yes', 'yes please', 'go on', 'continue', 'and')
    
    if not is_vague or not history:
        return user_query
    
    for msg in reversed(history):
        if msg.get('role') == 'user':
            prev = msg['content'].strip()
            if len(prev) > 5 and prev.lower() not in [v for v in vague_patterns]:
                resolved = f"{user_query} (regarding: {prev})"
                logger.info(f"Resolved vague query: '{user_query}' → '{resolved}'")
                return resolved
    
    return user_query


def handle_multi_intent_query(intents: list, user_query: str, ctx: RequestContext, history: list = None) -> str:
    if history is None:
        history = []
    
    contact_category = classify_contact_intent(user_query, intents)
    include_contact = should_include_contact(contact_category)
    
    logger.info(f"[{ctx.request_id}] Multi-intent contact category: {contact_category}")
    
    if len(intents) == 1:
        return handle_single_intent(intents[0], user_query, ctx, contact_category, history)
    
    all_contexts = []
    for intent in intents[:3]:
        retrieval_result = retrieve_with_quality_check(intent, user_query, ctx)
        if retrieval_result["success"]:
            all_contexts.append(retrieval_result["context"])
    
    if not all_contexts:
        if include_contact:
            return (
                "I'm not sure about that specific detail. Our team can give you accurate information.\n\n"
                f"📞 {FIRM_CONTACT['phone']}\n"
                f"📧 {FIRM_CONTACT['email']}"
            )
        return "I don't have that specific information. Could you rephrase your question or ask about something else?"
    
    seen = set()
    unique_parts = []
    for ctx_text in all_contexts:
        for part in ctx_text.split("\n\n"):
            part_key = part.strip()[:100]
            if part_key not in seen:
                seen.add(part_key)
                unique_parts.append(part.strip())
    
    combined_context = "\n\n".join(unique_parts)
    combined_intent = " + ".join(intents[:3])
    
    system_prompt = build_rag_system_prompt(combined_context, combined_intent, include_contact)
    
    try:
        messages = _build_chat_messages(system_prompt, user_query, history)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3 if any(i in ('phone', 'email', 'address', 'contact_methods', 'pricing') for i in intents) else 0.6,
            max_tokens=250,
        )
        
        answer = response.choices[0].message.content
        formatted = format_response_by_intent(intents[0], answer, user_query)
        
        if not include_contact:
            formatted = strip_contact_info(formatted)
        
        logger.info(f"[{ctx.request_id}] Combined response length: {len(formatted)} chars")
        return formatted
    
    except Exception as e:
        logger.error(f"[{ctx.request_id}] LLM error in multi-intent: {str(e)}")
        if include_contact:
            return (
                "I'm having a technical issue. Please try again or contact us directly.\n\n"
                f"📞 {FIRM_CONTACT['phone']}\n"
                f"📧 {FIRM_CONTACT['email']}"
            )
        return "I'm having a technical issue. Please try again in a moment."


def _is_conversational(user_query: str) -> tuple:
    """Detect conversational/small-talk messages and return (category, match) or (None, None)."""
    q = user_query.lower().strip().rstrip('!?.,')

    greeting_exact = {
        'hi', 'hello', 'hey', 'hola', 'howdy', 'yo', 'sup',
        'hii', 'hiii', 'heya', 'heyo', 'ello', 'helo',
        'good morning', 'good afternoon', 'good evening', 'good day',
        'morning', 'afternoon', 'evening',
        'greetings', 'salutations', 'whats up', "what's up",
        'wassup', 'wazzup', 'hey hey', 'hi hi',
    }
    if q in greeting_exact:
        return ('greeting', q)
    greeting_startswith = [
        'hi ', 'hey ', 'hello ', 'good morning', 'good afternoon',
        'good evening', 'good day', 'howdy', 'greetings',
        'hola ', 'yo ',
    ]
    if any(q.startswith(g) for g in greeting_startswith):
        return ('greeting', q)

    farewell_exact = {
        'bye', 'goodbye', 'good bye', 'see you', 'see ya',
        'later', 'take care', 'cya', 'ttyl', 'peace',
        'gotta go', 'got to go', 'im leaving', "i'm leaving",
        'catch you later', 'until next time',
    }
    if q in farewell_exact:
        return ('farewell', q)
    farewell_starts = ['bye ', 'goodbye ', 'see you ', 'take care', 'have a good', 'have a great', 'have a nice']
    if any(q.startswith(f) for f in farewell_starts):
        return ('farewell', q)

    if any(t in q for t in ['thank', 'thanks', 'thx', 'appreciate', 'grateful']):
        return ('thanks', q)

    ack_exact = {
        'ok', 'okay', 'sure', 'cool', 'great', 'got it', 'noted',
        'understood', 'alright', 'right', 'fine', 'perfect',
        'awesome', 'nice', 'wonderful', 'sounds good', 'fair enough',
        'makes sense', 'i see', 'ah ok', 'oh ok', 'oh okay',
        'k', 'kk', 'yep', 'yup', 'yes', 'yeah', 'ya',
        'no problem', 'no worries', 'all good',
    }
    if q in ack_exact:
        return ('acknowledgment', q)

    how_phrases = ['how are you', "how's it going", 'how is it going',
                   'how are you doing', 'how do you do', "how's everything",
                   'how you doing', "how's your day", 'how is your day',
                   'you good', 'are you good', 'are you well']
    if any(h in q for h in how_phrases):
        return ('how_are_you', q)

    import re as _re
    identity_patterns = [
        r'\bwho are you\b', r'\bwhat are you\b(?!r)',
        r'\byour name\b',
        r"\bwhat'?s your name\b", r'\bwhat is your name\b',
        r'\btell me about yourself\b', r'\bintroduce yourself\b',
        r'\bwho am i talking to\b', r'\bwho am i speaking with\b',
        r'\bare you a bot\b', r'\bare you real\b', r'\bare you human\b',
        r'\bare you ai\b', r'\bare you a robot\b', r'\bare you a person\b',
    ]
    if any(_re.search(p, q) for p in identity_patterns):
        return ('identity', q)

    cap_patterns = [
        r'\bwhat can you do\b', r'\bwhat do you do\b',
        r'\bwhat can you help\b', r'\bhow can you help me\b',
        r'\bwhat are you able\b', r'\bwhat should i ask\b',
        r'\bwhat can i ask\b', r'\bwhat do you know\b',
        r'\bwhat topics\b', r'\bwhat questions can\b',
    ]
    if q.strip() == 'help me' or any(_re.search(p, q) for p in cap_patterns):
        return ('capabilities', q)

    return (None, None)


def _conversational_response(category: str) -> str:
    """Return a natural, human-like response for conversational messages."""
    import random
    from datetime import datetime

    hour = datetime.now().hour
    if hour < 12:
        time_greeting = "Good morning"
    elif hour < 17:
        time_greeting = "Good afternoon"
    else:
        time_greeting = "Good evening"

    responses = {
        'greeting': [
            f"{time_greeting}! How can I help you today?",
            f"{time_greeting}! Welcome to Cook County LLC. What can I help you with?",
            "Hey there! What can I do for you today?",
            "Hi! Welcome — feel free to ask me anything about our services.",
            f"{time_greeting}! I'm here to help. What's on your mind?",
        ],
        'farewell': [
            "Take care! Don't hesitate to come back if you need anything.",
            "Goodbye! Wishing you all the best.",
            "Have a great day! We're here whenever you need us.",
            "See you next time! Feel free to reach out anytime.",
            "Bye for now! Hope I was able to help.",
        ],
        'thanks': [
            "You're welcome! Let me know if there's anything else I can help with.",
            "Happy to help! Feel free to ask if anything else comes up.",
            "Anytime! I'm here if you need anything else.",
            "Glad I could help! Don't hesitate to reach out again.",
            "Of course! Is there anything else on your mind?",
        ],
        'acknowledgment': [
            "Got it! Let me know if you have any other questions.",
            "Sounds good! I'm here if you need anything else.",
            "Alright! Feel free to ask me anything anytime.",
            "Great! Don't hesitate to reach out if something else comes up.",
        ],
        'how_are_you': [
            "I'm doing great, thanks for asking! How can I help you today?",
            "All good on my end! What can I do for you?",
            "Doing well, thank you! What's on your mind?",
            "I'm here and ready to help! What would you like to know?",
        ],
        'identity': [
            "I'm the virtual assistant for Cook County LLC. I can help answer questions about our legal services, walk you through what we offer, or point you in the right direction. What would you like to know?",
            "I'm Cook County LLC's assistant — think of me as your first stop for questions about our firm, our services, or how we can help with your situation.",
            "I'm here on behalf of Cook County LLC to help answer your questions. Whether it's about property tax appeals, estate planning, or any of our other services — just ask!",
        ],
        'capabilities': [
            "I can help with questions about our legal services — things like property tax appeals, estate planning, business law, real estate, and more. I can also help you find contact information or point you toward scheduling a consultation. What would you like to know?",
            "Great question! I'm here to answer questions about Cook County LLC's services, share details about how we can help with your legal needs, and connect you with our team. Just ask away!",
            "I know quite a bit about our firm's services — from tax law and IRS representation to estate planning and real estate. Feel free to ask me anything and I'll do my best to help!",
        ],
    }

    options = responses.get(category, [])
    return random.choice(options) if options else "How can I help you today?"


def handle_single_intent(intent: str, user_query: str, ctx: RequestContext, contact_category: str = None, history: list = None) -> str:
    if history is None:
        history = []
    if contact_category is None:
        contact_category = classify_contact_intent(user_query, [intent])
    
    include_contact = should_include_contact(contact_category)
    
    logger.info(f"[{ctx.request_id}] Contact category: {contact_category}, Include contact: {include_contact}")
    
    if intent == 'appointment_booking':
        return (
            "I can't schedule appointments directly, but our team would be happy to set one up for you!\n\n"
            f"📞 {FIRM_CONTACT['phone']}\n"
            f"📧 {FIRM_CONTACT['email']}"
        )
    
    if intent in ['phone', 'email', 'address', 'contact_methods']:
        include_contact = True
    
    conv_category, _ = _is_conversational(user_query)
    if conv_category:
        logger.info(f"[{ctx.request_id}] Conversational: {conv_category}")
        return _conversational_response(conv_category)
    
    
    retrieval_query_text = _resolve_vague_query(user_query, history)
    
    retrieval_result = retrieve_with_quality_check(intent, retrieval_query_text, ctx)
    
    if not retrieval_result["success"]:
        reason = retrieval_result["metadata"].get("reason", "unknown")
        logger.warning(f"[{ctx.request_id}] Retrieval failed: {reason}")
        
        if include_contact:
            return (
                "I'm not sure about that specific detail. Our team can give you accurate information.\n\n"
                f"📞 {FIRM_CONTACT['phone']}\n"
                f"📧 {FIRM_CONTACT['email']}"
            )
        else:
            return "I don't have specific information about that right now. Could you try rephrasing, or ask about our services?"
    
    context = retrieval_result["context"]
    
    system_prompt = build_rag_system_prompt(context, intent, include_contact)
    
    try:
        messages = _build_chat_messages(system_prompt, user_query, history)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3 if intent in ('phone', 'email', 'address', 'contact_methods', 'pricing') else 0.6,
            max_tokens=250,
        )
        
        answer = response.choices[0].message.content
        
        formatted = format_response_by_intent(intent, answer, user_query)
        
        if not include_contact:
            formatted = strip_contact_info(formatted)
        
        logger.info(f"[{ctx.request_id}] Response length: {len(formatted)} chars")
        
        return formatted
    
    except Exception as e:
        logger.error(f"[{ctx.request_id}] LLM error: {str(e)}\n{traceback.format_exc()}")
        
        if include_contact:
            return (
                "I'm having a technical issue. Please try again or contact us directly.\n\n"
                f"📞 {FIRM_CONTACT['phone']}\n"
                f"📧 {FIRM_CONTACT['email']}"
            )
        else:
            return "I'm having a technical issue. Please try again in a moment."


@app.route('/')
def home():
    doc_count = collection.count()
    return render_template('index.html', doc_count=doc_count)


@app.route('/documents')
def documents():
    doc_count = collection.count()
    return render_template('documents.html', doc_count=doc_count)


@app.route('/chat', methods=['POST'])
def chat():
    ctx = RequestContext()
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        history = data.get('history', [])
        
        if not user_message:
            return jsonify({'response': 'Please enter a message.'})
        
        history = history[-10:] if history else []
        
        logger.info(f"[{ctx.request_id}] User query: '{user_message[:100]}' (history: {len(history)} msgs)")
        
        intents = detect_intents(user_message)
        logger.info(f"[{ctx.request_id}] Detected intents: {intents}")
        
        response = handle_multi_intent_query(intents, user_message, ctx, history)
        
        logger.info(f"[{ctx.request_id}] Complete in {ctx.elapsed()}s")
        
        return jsonify({'response': response})
    
    except Exception as e:
        logger.error(f"[{ctx.request_id}] Fatal error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'response': "I apologize, but I'm experiencing technical difficulties. Please try again or contact the office directly."
        }), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'documents': collection.count(),
        'model': LLM_MODEL
    })


@app.route('/debug', methods=['POST'])
def debug_query():
    ctx = RequestContext()
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        intent = data.get('intent', 'general')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        retrieval_result = retrieve_with_quality_check(intent, query, ctx)
        
        return jsonify({
            'request_id': ctx.request_id,
            'success': retrieval_result['success'],
            'context_length': len(retrieval_result['context']),
            'metadata': retrieval_result['metadata'],
            'context_preview': retrieval_result['context'][:500] if retrieval_result['context'] else None
        })
    
    except Exception as e:
        logger.error(f"[{ctx.request_id}] Debug error: {e}")
        return jsonify({'error': str(e)}), 500


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/chatbots/<chatbot_id>/documents', methods=['GET'])
def get_documents(chatbot_id):
    db = SessionLocal()
    try:
        docs = db.query(Document).filter(Document.chatbot_id == chatbot_id).all()
        return jsonify({'documents': [doc.to_dict() for doc in docs]})
    finally:
        db.close()


@app.route('/api/chatbots/<chatbot_id>/documents', methods=['POST'])
def upload_document(chatbot_id):
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    doc_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    upload_dir = os.path.join(UPLOAD_FOLDER, doc_id)
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)
    
    file_size = os.path.getsize(file_path)
    
    db = SessionLocal()
    try:
        doc = Document(
            id=doc_id,
            chatbot_id=chatbot_id,
            filename=filename,
            storage_path=file_path,
            file_size=file_size,
            status=DocumentStatus.PROCESSING
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(pages)
            
            documents = []
            ids = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                if chunk.page_content.strip():
                    documents.append(chunk.page_content)
                    ids.append(f"doc_{doc_id}_{i}")
                    metadatas.append({
                        'document_id': doc_id,
                        'filename': filename,
                        'page': chunk.metadata.get('page', 0)
                    })
            
            with contextlib.redirect_stderr(_stderr_suppressor):
                collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
            
            doc.status = DocumentStatus.TRAINED
            doc.chunk_count = len(documents)
            db.commit()
            
        except Exception as e:
            doc.status = DocumentStatus.FAILED
            doc.error_message = str(e)
            db.commit()
            logger.error(f"Training failed for {filename}: {e}")
        
        db.refresh(doc)
        return jsonify({'document': doc.to_dict()}), 201
    finally:
        db.close()


@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
        
        if os.path.exists(doc.storage_path):
            os.remove(doc.storage_path)
            parent_dir = os.path.dirname(doc.storage_path)
            if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
        
        with contextlib.redirect_stderr(_stderr_suppressor):
            try:
                ids_to_delete = [f"doc_{doc_id}_{i}" for i in range(doc.chunk_count or 100)]
                collection.delete(ids=ids_to_delete)
            except Exception:
                pass
        
        db.delete(doc)
        db.commit()
        return jsonify({'success': True})
    finally:
        db.close()


@app.route('/api/documents/<doc_id>/train', methods=['POST'])
def train_document(doc_id):
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
        
        doc.status = DocumentStatus.PROCESSING
        db.commit()
        
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            loader = PyPDFLoader(doc.storage_path)
            pages = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(pages)
            
            documents = []
            ids = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                if chunk.page_content.strip():
                    documents.append(chunk.page_content)
                    ids.append(f"doc_{doc_id}_{i}")
                    metadatas.append({
                        'document_id': doc_id,
                        'filename': doc.filename,
                        'page': chunk.metadata.get('page', 0)
                    })
            
            with contextlib.redirect_stderr(_stderr_suppressor):
                collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
            
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.status = DocumentStatus.TRAINED
                doc.chunk_count = len(documents)
                db.commit()
            
            return jsonify({'success': True, 'chunks': len(documents)})
            
        except Exception as e:
            db.rollback()
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.status = DocumentStatus.FAILED
                doc.error_message = str(e)
                db.commit()
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()


@app.route('/api/documents/stats', methods=['GET'])
def get_stats():
    db = SessionLocal()
    try:
        total_docs = db.query(Document).count()
        trained_docs = db.query(Document).filter(Document.status == DocumentStatus.TRAINED).count()
        total_embeddings = collection.count()
        
        return jsonify({
            'total_documents': total_docs,
            'trained_documents': trained_docs,
            'total_embeddings': total_embeddings,
            'orphaned_embeddings': 0
        })
    finally:
        db.close()


@app.route('/api/documents/reset-all', methods=['POST'])
def reset_all_documents():
    db = SessionLocal()
    try:
        docs = db.query(Document).all()
        deleted_documents = len(docs)
        
        for doc in docs:
            if os.path.exists(doc.storage_path):
                os.remove(doc.storage_path)
                parent_dir = os.path.dirname(doc.storage_path)
                if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
            db.delete(doc)
        
        deleted_embeddings = collection.count()
        
        with contextlib.redirect_stderr(_stderr_suppressor):
            all_ids = collection.get()['ids']
            if all_ids:
                collection.delete(ids=all_ids)
        
        db.commit()
        
        return jsonify({
            'success': True,
            'deleted_documents': deleted_documents,
            'deleted_embeddings': deleted_embeddings
        })
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()


if __name__ == '__main__':
    init_db()
    print(f"📚 Loaded {collection.count()} documents from ChromaDB")
    print(f"🚀 Starting RAG-First Legal Chatbot...")
    print(f"🌐 Open http://localhost:5001 in your browser")
    app.run(debug=False, host='0.0.0.0', port=5001)
