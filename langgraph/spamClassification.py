from typing import Any, TypedDict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

import warnings

warnings.filterwarnings("ignore")

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)


class EmailState(TypedDict):
    email: dict[str, Any]
    isSpam: Optional[bool]
    spamReason: Optional[str]
    messages: list[dict[str, str]]
    emailCategory: Optional[str]
    emailDraft: Optional[str]


def readEmail(state: EmailState):
    email = state["email"]

    print("Reading email...")

    return {}


def classifyMail(state: EmailState):
    email = state["email"]

    # Prepare prompt for the LLM
    prompt = f"""
    Analyze this email and determine if it is spam or legitimate.

    Email:
    From: {email["sender"]}
    Subject: {email["subject"]}
    Body: {email["body"]}

    First, determine if this email is spam. If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).
    """

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    responseText = response.content

    isSpam = "spam" in responseText.lower() and "not spam" not in responseText.lower()

    spamReason = None

    if isSpam and "reason" in responseText.lower():
        spamReason = responseText.lower().split("reason", 1)[1].strip()

    emailCategory = None

    if not isSpam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]

        for category in categories:
            if category in responseText.lower():
                emailCategory = category
                break

    newMessages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": responseText},
    ]

    return {
        "messages": newMessages,
        "isSpam": isSpam,
        "spamReason": spamReason,
        "emailCategory": emailCategory,
    }


def handleSpam(state: EmailState):
    """Handle an email identified as spam."""

    print(f"Email marked as spam. Reason: {state['spamReason']}")

    print("The email has been moved to the spam folder.")

    return {}


def draftResponse(state: EmailState):
    """Draft a preliminary response for a legitimate email."""

    email = state["email"]
    category = state["emailCategory"] or "general"

    # Prepare prompt for the LLM
    prompt = f"""
    Draft a polite and professional preliminary response to this email.

    Email:
    From: {email["sender"]}
    Subject: {email["subject"]}
    Body: {email["body"]}

    This email has been categorized as: {category}

    Draft a brief, professional response that can be reviewed
    and personalized before sending.
    """

    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    # Update messages for tracking
    newMessages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content},
    ]

    return {"emailDraft": response.content, "messages": newMessages}


def notifyUser(state: EmailState):
    """Notify the user about the email and present the draft."""

    email = state["email"]

    print("\n" + "=" * 50)
    print(f"Email received from: {email['sender']}")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['emailCategory']}")

    print("\nDraft response:")
    print(state["emailDraft"])

    print("=" * 50 + "\n")

    return {}


def routeEmail(state: EmailState) -> str:
    """Determine the next step based on spam classification."""

    if state["isSpam"]:
        return "Handle spam"

    return "Draft response"


# Create the graph
graph = StateGraph(EmailState)

# Add nodes
graph.add_node("Read email", readEmail)
graph.add_node("Classify mail", classifyMail)
graph.add_node("Handle spam", handleSpam)
graph.add_node("Draft response", draftResponse)
graph.add_node("Notify user", notifyUser)

# Define workflow
graph.add_edge(START, "Read email")
graph.add_edge("Read email", "Classify mail")

graph.add_conditional_edges("Classify mail", routeEmail)

graph.add_edge("Handle spam", END)

graph.add_edge("Draft response", "Notify user")

graph.add_edge("Notify user", END)

# Compile the graph
result = graph.compile()


# Example legitimate email
legitimateEmail = {
    "sender": "john.smith@example.com",
    "subject": "Question about your services",
    "body": """
    Dear Sir/Madam,

    I was referred to you by a colleague and I'm interested
    in learning more about your consulting services.

    Could we schedule a call next week?

    Best regards,
    John Smith
    """,
}


# Example spam email
spamEmail = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": """
    CONGRATULATIONS!

    You have been selected as the winner of our international lottery!

    To claim your $5,000,000 prize, please send us your bank
    details and a processing fee of $100.
    """,
}


# Process legitimate email
print("\nProcessing legitimate email...")

legitimateResult = result.invoke(
    {
        "email": legitimateEmail,
        "isSpam": None,
        "spamReason": None,
        "emailCategory": None,
        "emailDraft": None,
        "messages": [],
    }
)


# Process spam email
print("\nProcessing spam email...")

spamResult = result.invoke(
    {
        "email": spamEmail,
        "isSpam": None,
        "spamReason": None,
        "emailCategory": None,
        "emailDraft": None,
        "messages": [],
    }
)
