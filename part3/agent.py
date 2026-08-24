import json
import os
import re
from typing import TypedDict, Optional

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from langgraph.graph import StateGraph, END


# ============================================================
# IMPORT RETRIEVER
# ============================================================

try:
    from .retrieve import retrieve_policy
except ImportError:
    from retrieve import retrieve_policy


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

POLICY_SIMILARITY_THRESHOLD = 0.40
RETURN_RISK_THRESHOLD = 0.47

RETURN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "part1",
    "models",
    "return_risk_model.pkl"
)

IMAGE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "part2",
    "models",
    "product_classifier.pt"
)

SAMPLE_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "part2",
    "data",
    "sample_images"
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Flipkart's support assistant.

Specific:
Answer only using supplied policy evidence or tool results.

Short:
Keep answers concise and directly useful.

Surround:
Treat retrieved policy chunks and tool results as authoritative.

Single:
Return exactly one JSON object containing:
answer, source, confidence.

Role:
You are a Flipkart order-support assistant.
"""


# ============================================================
# GRAPH STATE
# ============================================================

class AgentState(TypedDict, total=False):

    messages: list

    query: str
    intent: str

    retrieved_chunks: list

    order_features: Optional[dict]
    image_path: Optional[str]

    tool_result: Optional[dict]

    response: Optional[dict]

    blocked: bool

    state_order_id: Optional[str]


# ============================================================
# PROMPT-INJECTION GUARDRAIL
# ============================================================

INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all rules",
    r"ignore the instructions",
    r"disregard previous instructions",
    r"forget your instructions",
    r"pretend you are",
    r"act as if you are",
]


def contains_prompt_injection(text):

    text_lower = text.lower()

    for pattern in INJECTION_PATTERNS:

        if re.search(
            pattern,
            text_lower
        ):
            return True

    return False


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def intent_node(state):

    query = state.get(
        "query",
        ""
    )

    if contains_prompt_injection(query):

        return {
            "intent": "blocked",
            "blocked": True
        }

    query_lower = query.lower()

    # --------------------------------------------------------
    # PRODUCT IMAGE
    # --------------------------------------------------------

    if (
        "image" in query_lower
        or "picture" in query_lower
        or "photo" in query_lower
        or "product category" in query_lower
        or "category is this product" in query_lower
    ):

        return {
            "intent": "product_category",
            "blocked": False
        }

    # --------------------------------------------------------
    # RETURN RISK
    # --------------------------------------------------------

    if (
        "return risk" in query_lower
        or "risk" in query_lower
        or "likely to return" in query_lower
        or "return probability" in query_lower
        or "risk bucket" in query_lower
        or "risk classification" in query_lower
    ):

        return {
            "intent": "return_risk",
            "blocked": False
        }

    # --------------------------------------------------------
    # DEFAULT = POLICY
    # --------------------------------------------------------

    return {
        "intent": "policy",
        "blocked": False
    }


# ============================================================
# POLICY RETRIEVAL
# ============================================================

def retrieval_node(state):

    query = state.get(
        "query",
        ""
    )

    results = retrieve_policy(
        query,
        top_k=5
    )

    return {
        "retrieved_chunks": results
    }


# ============================================================
# RETURN-RISK TOOL
# ============================================================

def check_return_risk(order_features: dict) -> dict:

    import joblib

    if not order_features:

        raise ValueError(
            "order_features are required for return-risk prediction."
        )

    model = joblib.load(
        RETURN_MODEL_PATH
    )

    # IMPORTANT:
    # The saved sklearn Pipeline expects a pandas DataFrame.
    # Passing [order_features] creates a 1D object array and
    # causes:
    #
    # ValueError: Expected 2D array, got 1D array instead.
    #
    # Therefore we create a one-row DataFrame.

    feature_order = [
        "product_category",
        "price_inr",
        "discount_pct",
        "payment_method",
        "customer_tenure_days",
        "num_previous_orders",
        "num_previous_returns",
        "delivery_distance_km",
        "delivery_days",
        "is_weekend_order",
        "rating_given"
    ]

    missing = [
        feature
        for feature in feature_order
        if feature not in order_features
    ]

    if missing:

        raise ValueError(
            "Missing required order features: "
            + ", ".join(missing)
        )

    input_data = pd.DataFrame(
        [
            {
                feature: order_features[feature]
                for feature in feature_order
            }
        ]
    )

    probability = float(
        model.predict_proba(
            input_data
        )[0][1]
    )

    # --------------------------------------------------------
    # Risk buckets
    # --------------------------------------------------------

    if probability < RETURN_RISK_THRESHOLD:

        bucket = "Low"

    elif probability < (
        RETURN_RISK_THRESHOLD + 0.15
    ):

        bucket = "Medium"

    else:

        bucket = "High"

    return {
        "predicted_return_probability": round(
            probability,
            4
        ),
        "risk_bucket": bucket,
        "t_rf": RETURN_RISK_THRESHOLD,
        "bucket_cut_points": {
            "low": (
                f"probability < "
                f"{RETURN_RISK_THRESHOLD:.4f}"
            ),
            "medium": (
                f"{RETURN_RISK_THRESHOLD:.4f} <= "
                f"probability < "
                f"{RETURN_RISK_THRESHOLD + 0.15:.4f}"
            ),
            "high": (
                f"probability >= "
                f"{RETURN_RISK_THRESHOLD + 0.15:.4f}"
            )
        }
    }


# ============================================================
# LOAD IMAGE MODEL
# ============================================================

def load_image_model():

    model = torch.load(
        IMAGE_MODEL_PATH,
        map_location="cpu"
    )

    # --------------------------------------------------------
    # Case 1:
    # Full PyTorch model was saved.
    # --------------------------------------------------------

    if hasattr(
        model,
        "eval"
    ):

        model.eval()

        return model

    # --------------------------------------------------------
    # Case 2:
    # State dictionary was saved.
    #
    # Recreate the architecture used by Part 2.
    # --------------------------------------------------------

    if isinstance(
        model,
        dict
    ):

        # Some checkpoints store state_dict under this key.
        if "state_dict" in model:

            state_dict = model[
                "state_dict"
            ]

        else:

            state_dict = model

        from torchvision.models import resnet18

        classifier = resnet18(
            weights=None
        )

        classifier.fc = torch.nn.Linear(
            classifier.fc.in_features,
            10
        )

        # Handle possible DataParallel prefixes.
        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith(
                "module."
            ):

                key = key[
                    len("module.") :
                ]

            cleaned_state_dict[
                key
            ] = value

        classifier.load_state_dict(
            cleaned_state_dict,
            strict=False
        )

        classifier.eval()

        return classifier

    raise TypeError(
        "Unsupported image model format: "
        + str(type(model))
    )


# ============================================================
# PRODUCT IMAGE CLASSIFIER
# ============================================================

def classify_product_image(
    image_path: str
) -> dict:

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            "Image not found: "
            + image_path
        )

    model = load_image_model()

    image = Image.open(
        image_path
    ).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(
            (224, 224)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])

    tensor = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    labels = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
    ]

    predicted_label = labels[
        int(
            prediction.item()
        )
    ]

    return {
        "predicted_category": predicted_label,
        "confidence": round(
            float(
                confidence.item()
            ),
            4
        ),
        "image_path": os.path.abspath(
            image_path
        )
    }


# ============================================================
# TOOL NODE
# ============================================================

def tool_node(state):

    intent = state.get(
        "intent"
    )

    # --------------------------------------------------------
    # RETURN RISK
    # --------------------------------------------------------

    if intent == "return_risk":

        features = state.get(
            "order_features"
        )

        # If a follow-up asks about the same order,
        # the previous state must be supplied.

        if not features:

            return {
                "tool_result": {
                    "error": (
                        "Order information is required "
                        "to calculate return risk."
                    )
                }
            }

        result = check_return_risk(
            features
        )

        return {
            "tool_result": result
        }

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if intent == "product_category":

        image_path = state.get(
            "image_path"
        )

        if not image_path:

            image_path = os.path.join(
                SAMPLE_IMAGE_DIR,
                "01_T-shirt_top.png"
            )

        result = classify_product_image(
            image_path
        )

        return {
            "tool_result": result
        }

    return {
        "tool_result": None
    }


# ============================================================
# RESPONSE NODE
# ============================================================

def response_node(state):

    # --------------------------------------------------------
    # BLOCKED
    # --------------------------------------------------------

    if state.get(
        "blocked"
    ):

        return {
            "response": {
                "answer": (
                    "I can't follow instructions that attempt "
                    "to override the support assistant's rules."
                ),
                "source": "policy_kb",
                "confidence": 1.0
            }
        }

    intent = state.get(
        "intent"
    )

    # ========================================================
    # POLICY
    # ========================================================

    if intent == "policy":

        results = state.get(
            "retrieved_chunks",
            []
        )

        if not results:

            return {
                "response": {
                    "answer": (
                        "I could not find sufficiently relevant "
                        "information in the policy knowledge base "
                        "to answer this question."
                    ),
                    "source": "policy_kb",
                    "confidence": 0.0
                }
            }

        best_score = float(
            results[0][
                "similarity_score"
            ]
        )

        # ----------------------------------------------------
        # Groundedness guardrail
        # ----------------------------------------------------

        if (
            best_score
            < POLICY_SIMILARITY_THRESHOLD
        ):

            return {
                "response": {
                    "answer": (
                        "I could not find sufficiently relevant "
                        "information in the policy knowledge base "
                        "to answer this question."
                    ),
                    "source": "policy_kb",
                    "confidence": round(
                        best_score,
                        4
                    )
                }
            }

        # ----------------------------------------------------
        # Select relevant evidence.
        #
        # Do not repeat chunks from the same document.
        # ----------------------------------------------------

        evidence = []

        seen_documents = set()

        for result in results:

            score = float(
                result[
                    "similarity_score"
                ]
            )

            document_id = result[
                "document_id"
            ]

            if (
                score >= POLICY_SIMILARITY_THRESHOLD
                and document_id not in seen_documents
            ):

                evidence.append(
                    result["text"]
                )

                seen_documents.add(
                    document_id
                )

            if len(evidence) >= 2:

                break

        return {
            "response": {
                "answer": " ".join(
                    evidence
                ),
                "source": "policy_kb",
                "confidence": round(
                    best_score,
                    4
                )
            }
        }

    # ========================================================
    # RETURN RISK
    # ========================================================

    if intent == "return_risk":

        result = state.get(
            "tool_result"
        )

        if not result or "error" in result:

            return {
                "response": {
                    "answer": (
                        result.get(
                            "error",
                            "Order information is required."
                        )
                        if result
                        else
                        "Order information is required."
                    ),
                    "source": "return_risk_tool",
                    "confidence": 0.0
                }
            }

        probability = float(
            result[
                "predicted_return_probability"
            ]
        )

        bucket = result[
            "risk_bucket"
        ]

        return {
            "response": {
                "answer": (
                    f"The predicted return probability is "
                    f"{probability:.4f}, giving this order a "
                    f"{bucket} return-risk classification. "
                    f"The risk buckets are anchored to "
                    f"t*_rf = {RETURN_RISK_THRESHOLD:.4f}."
                ),
                "source": "return_risk_tool",
                "confidence": probability
            }
        }

    # ========================================================
    # PRODUCT CATEGORY
    # ========================================================

    if intent == "product_category":

        result = state.get(
            "tool_result"
        )

        if not result:

            return {
                "response": {
                    "answer": (
                        "I could not classify the product image."
                    ),
                    "source": "image_classifier_tool",
                    "confidence": 0.0
                }
            }

        return {
            "response": {
                "answer": (
                    f"The product is classified as "
                    f"{result['predicted_category']} with "
                    f"{result['confidence']:.4f} confidence."
                ),
                "source": "image_classifier_tool",
                "confidence": result[
                    "confidence"
                ]
            }
        }

    return {
        "response": {
            "answer": (
                "I could not determine the request type."
            ),
            "source": "policy_kb",
            "confidence": 0.0
        }
    }


# ============================================================
# ROUTING
# ============================================================

def route_after_intent(state):

    if state.get(
        "blocked"
    ):

        return "response"

    intent = state.get(
        "intent"
    )

    if intent == "policy":

        return "retrieval"

    if intent in [
        "return_risk",
        "product_category"
    ]:

        return "tool"

    return "response"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(
    AgentState
)

graph_builder.add_node(
    "intent",
    intent_node
)

graph_builder.add_node(
    "retrieval",
    retrieval_node
)

graph_builder.add_node(
    "tool",
    tool_node
)

graph_builder.add_node(
    "response",
    response_node
)

graph_builder.set_entry_point(
    "intent"
)

graph_builder.add_conditional_edges(
    "intent",
    route_after_intent,
    {
        "retrieval": "retrieval",
        "tool": "tool",
        "response": "response"
    }
)

graph_builder.add_edge(
    "retrieval",
    "response"
)

graph_builder.add_edge(
    "tool",
    "response"
)

graph_builder.add_edge(
    "response",
    END
)

graph = graph_builder.compile()


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(
    query,
    order_features=None,
    image_path=None,
    messages=None,
    state_order_id=None
):

    initial_state = {
        "query": query,
        "messages": messages or [],
        "order_features": order_features,
        "image_path": image_path,
        "state_order_id": state_order_id
    }

    result = graph.invoke(
        initial_state
    )

    return result


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "Flipkart Order Intelligence Support Agent"
    )
    print(
        "MOCK_LLM MODE"
    )
    print("=" * 70)

    query = input(
        "\nEnter your query: "
    )

    result = run_agent(
        query
    )

    print(
        "\nIntent:"
    )

    print(
        result.get(
            "intent"
        )
    )

    if result.get(
        "retrieved_chunks"
    ):

        print(
            "\nRetrieved Policy Evidence:"
        )

        for item in result[
            "retrieved_chunks"
        ]:

            print(
                f"{item['document_id']} | "
                f"{item['similarity_score']:.4f}"
            )

    if result.get(
        "tool_result"
    ):

        print(
            "\nTool Result:"
        )

        print(
            json.dumps(
                result[
                    "tool_result"
                ],
                indent=4
            )
        )

    print(
        "\nFinal Response:"
    )

    print(
        json.dumps(
            result[
                "response"
            ],
            indent=4
        )
    )

    print(
        "\nConversation state available:",
        bool(
            result.get(
                "order_features"
            )
            or result.get(
                "state_order_id"
            )
        )
    )