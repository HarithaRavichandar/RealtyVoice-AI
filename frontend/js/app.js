/**
 * RealtyVoice AI
 * Frontend integration with FastAPI backend.
 *
 * Features:
 * - Backend health check
 * - Conversation creation and session restoration
 * - Agent message exchange
 * - Expired-session recovery with one retry
 * - Safe DOM rendering
 * - Customer requirements display
 * - Matched property cards
 * - Suggestion buttons
 * - Callback status cards (success / pending)
 * - Typing indicator while waiting for agent
 * - Error toast notifications
 * - Auto-resize textarea
 * - Enter / Shift+Enter keyboard handling
 */

// ==================================================
// 1. DOM REFERENCES
// ==================================================

const statusEl = document.getElementById("conn-status");
const sendBtn = document.getElementById("send-btn");
const msgInput = document.getElementById("msg-input");
const chatContainer = document.getElementById("chat");
const welcomeSection = document.getElementById("welcome");

const requirementsPanel = document.getElementById("requirements");
const requirementsList = document.getElementById("requirements-list");

const propertiesPanel = document.getElementById("properties");
const propertiesList = document.getElementById("properties-list");

const SESSION_STORAGE_KEY = "session_id";

// ==================================================
// 2. APPLICATION STATE
// ==================================================

let sessionId = null;
let busy = false;

// ==================================================
// 3. FORMATTING HELPERS
// ==================================================

const inrFormatter = new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0
});

function formatPrice(value) {
    if (value === null || value === undefined || value === "") {
        return null;
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return String(value);
    }

    return inrFormatter.format(number);
}

function hasValue(value) {
    if (value === null || value === undefined || value === "") {
        return false;
    }

    if (Array.isArray(value)) {
        return value.length > 0;
    }

    if (typeof value === "string") {
        return value.trim().length > 0;
    }

    return true;
}

function formatDate(value) {
    if (!hasValue(value)) return null;

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return String(value);
    }

    return new Intl.DateTimeFormat("en-IN", {
        day: "numeric",
        month: "short",
        year: "numeric"
    }).format(date);
}

// ==================================================
// 4. GENERAL UI HELPERS
// ==================================================

function setStatus(message, connected = false) {
    if (!statusEl) return;

    statusEl.textContent = message;
    statusEl.style.color = connected ? "#4caf50" : "#f44336";
}

function updateSendButton() {
    if (!sendBtn || !msgInput) return;

    const hasText = msgInput.value.trim().length > 0;

    sendBtn.disabled = busy || !sessionId || !hasText;
}

function showChat() {
    if (chatContainer) {
        chatContainer.hidden = false;
        chatContainer.removeAttribute("hidden");
    }

    if (welcomeSection) {
        welcomeSection.hidden = true;
        welcomeSection.classList.add("hidden");
    }
}

function createTextElement(tag, className, text) {
    const element = document.createElement(tag);

    if (className) {
        element.className = className;
    }

    element.textContent = String(text);

    return element;
}

// ==================================================
// 5. CHAT MESSAGE RENDERING
// ==================================================

function renderMessage(message, sender = "assistant") {
    if (!chatContainer || !hasValue(message)) return;

    showChat();

    const bubble = document.createElement("div");
    bubble.className = sender === "user" ? "bubble user" : "bubble assistant";

    if (sender === "assistant") {
        // Parse markdown safely without using innerHTML.
        const contentEl = parseMarkdown(message);
        bubble.appendChild(contentEl);
    } else {
        // User messages are plain text.
        bubble.textContent = String(message);
    }

    chatContainer.appendChild(bubble);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Safe, dependency-free Markdown renderer.
// All generated elements are created through DOM APIs; untrusted content is
// assigned only through textContent or text nodes. No innerHTML is used.
function parseMarkdown(markdown) {
    const container = document.createElement("div");
    const lines = String(markdown).split(/\r?\n/);

    let index = 0;

    while (index < lines.length) {
        const line = lines[index].trim();

        if (!line) {
            index += 1;
            continue;
        }

        // Markdown table: header row followed by a separator row.
        if (
            line.includes("|") &&
            index + 1 < lines.length &&
            /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[index + 1])
        ) {
            const table = document.createElement("table");
            const thead = document.createElement("thead");
            const tbody = document.createElement("tbody");
            const headerRow = document.createElement("tr");

            splitTableRow(line).forEach((cellText) => {
                const th = document.createElement("th");
                th.appendChild(parseInline(cellText));
                headerRow.appendChild(th);
            });

            thead.appendChild(headerRow);
            table.append(thead, tbody);
            index += 2;

            while (
                index < lines.length &&
                lines[index].includes("|") &&
                lines[index].trim()
            ) {
                const row = document.createElement("tr");

                splitTableRow(lines[index]).forEach((cellText) => {
                    const td = document.createElement("td");
                    td.appendChild(parseInline(cellText));
                    row.appendChild(td);
                });

                tbody.appendChild(row);
                index += 1;
            }

            container.appendChild(table);
            continue;
        }

        // Headings.
        const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);

        if (headingMatch) {
            const heading = document.createElement(
                `h${headingMatch[1].length}`
            );

            heading.appendChild(parseInline(headingMatch[2]));
            container.appendChild(heading);
            index += 1;
            continue;
        }

        // Consecutive ordered or unordered list items.
        const unorderedMatch = line.match(/^[-*+]\s+(.+)$/);
        const orderedMatch = line.match(/^\d+[.)]\s+(.+)$/);

        if (unorderedMatch || orderedMatch) {
            const list = document.createElement(
                unorderedMatch ? "ul" : "ol"
            );

            while (index < lines.length) {
                const current = lines[index].trim();

                const itemMatch = unorderedMatch
                    ? current.match(/^[-*+]\s+(.+)$/)
                    : current.match(/^\d+[.)]\s+(.+)$/);

                if (!itemMatch) break;

                const item = document.createElement("li");
                item.appendChild(parseInline(itemMatch[1]));
                list.appendChild(item);
                index += 1;
            }

            container.appendChild(list);
            continue;
        }

        // Paragraph with soft line breaks preserved as <br>.
        const paragraph = document.createElement("p");
        let firstLine = true;

        while (index < lines.length && lines[index].trim()) {
            const current = lines[index].trim();

            if (
                /^(#{1,6})\s+/.test(current) ||
                /^[-*+]\s+/.test(current) ||
                /^\d+[.)]\s+/.test(current)
            ) {
                break;
            }

            if (
                current.includes("|") &&
                index + 1 < lines.length &&
                /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[index + 1])
            ) {
                break;
            }

            if (!firstLine) {
                paragraph.appendChild(document.createElement("br"));
            }

            paragraph.appendChild(parseInline(current));
            firstLine = false;
            index += 1;
        }

        if (paragraph.childNodes.length) {
            container.appendChild(paragraph);
        } else {
            // Defensive progress guard for malformed input.
            index += 1;
        }
    }

    return container;
}

function splitTableRow(row) {
    let value = row.trim();

    if (value.startsWith("|")) {
        value = value.slice(1);
    }

    if (value.endsWith("|")) {
        value = value.slice(0, -1);
    }

    return value.split("|").map((cell) => cell.trim());
}

// Parse a small safe subset of inline Markdown.
function parseInline(text) {
    const fragment = document.createDocumentFragment();
    const input = String(text);

    const tokenPattern =
        /(\*\*[^*]+\*\*|__[^_]+__|(?<!\*)\*[^*\n]+\*(?!\*)|(?<!_)_[^_\n]+_(?!_)|`[^`]+`|\[([^\]]+)\]\(([^)\s]+)\))/g;

    let lastIndex = 0;
    let match;

    while ((match = tokenPattern.exec(input)) !== null) {
        if (match.index > lastIndex) {
            fragment.appendChild(
                document.createTextNode(
                    input.slice(lastIndex, match.index)
                )
            );
        }

        const token = match[0];
        let element = null;

        if (token.startsWith("**") || token.startsWith("__")) {
            element = document.createElement("strong");
            element.textContent = token.slice(2, -2);

        } else if (
            (token.startsWith("*") && token.endsWith("*")) ||
            (token.startsWith("_") && token.endsWith("_"))
        ) {
            element = document.createElement("em");
            element.textContent = token.slice(1, -1);

        } else if (token.startsWith("`")) {
            element = document.createElement("code");
            element.textContent = token.slice(1, -1);

        } else if (token.startsWith("[")) {
            const label = match[2];
            const href = match[3];

            if (/^(https?:\/\/|mailto:)/i.test(href)) {
                element = document.createElement("a");
                element.textContent = label;
                element.setAttribute("href", href);
                element.setAttribute("target", "_blank");
                element.setAttribute("rel", "noopener noreferrer");
            } else {
                element = document.createTextNode(label);
            }
        }

        if (element) {
            fragment.appendChild(element);
        } else {
            fragment.appendChild(document.createTextNode(token));
        }

        lastIndex = tokenPattern.lastIndex;
    }

    if (lastIndex < input.length) {
        fragment.appendChild(
            document.createTextNode(input.slice(lastIndex))
        );
    }

    return fragment;
}

function showError(message) {
    renderMessage(message, "assistant");
}

// ==================================================
// 6. TYPING INDICATOR
// ==================================================

let typingBubble = null;

function showTypingIndicator() {
    if (!chatContainer) return;

    showChat();

    typingBubble = document.createElement("div");
    typingBubble.className = "bubble assistant typing-indicator";
    typingBubble.setAttribute("aria-label", "Assistant is typing");

    for (let i = 0; i < 3; i++) {
        const dot = document.createElement("span");
        dot.className = "typing-dot";
        typingBubble.appendChild(dot);
    }

    chatContainer.appendChild(typingBubble);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function hideTypingIndicator() {
    if (typingBubble && typingBubble.parentNode) {
        typingBubble.parentNode.removeChild(typingBubble);
    }
    typingBubble = null;
}

// ==================================================
// 7. ERROR TOAST
// ==================================================

function showToast(message, type = "error") {
    // Remove any existing toast.
    const existing = document.querySelector(".toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.setAttribute("role", "alert");
    toast.textContent = message;

    document.body.appendChild(toast);

    // Trigger the enter animation on the next frame.
    requestAnimationFrame(() => {
        toast.classList.add("toast-visible");
    });

    // Auto-dismiss after 5 seconds.
    setTimeout(() => {
        toast.classList.remove("toast-visible");
        toast.addEventListener("transitionend", () => toast.remove(), { once: true });
        // Fallback removal if transitionend doesn't fire.
        setTimeout(() => toast.remove(), 400);
    }, 5000);
}

// ==================================================
// 8. AUTO-RESIZE TEXTAREA
// ==================================================

function autoResizeInput() {
    if (!msgInput) return;

    // Reset to single-row height so scrollHeight reflects content.
    msgInput.style.height = "auto";

    // Cap at roughly 6 lines (~9rem).
    const maxHeight = 144;
    const newHeight = Math.min(msgInput.scrollHeight, maxHeight);

    msgInput.style.height = `${newHeight}px`;
}

// ==================================================
// 6. BACKEND HEALTH CHECK
// ==================================================

async function checkBackend() {
    try {
        const response = await fetch("/health");

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        setStatus("Connected", true);
        return true;

    } catch (error) {
        console.error("Backend health check failed:", error);

        setStatus("Backend unavailable", false);
        return false;
    }
}

// ==================================================
// 7. CONVERSATION MANAGEMENT
// ==================================================

async function createConversation() {
    const response = await fetch("/api/conversations", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            channel: "browser_text"
        })
    });

    if (!response.ok) {
        throw new Error(
            `Conversation creation failed: HTTP ${response.status}`
        );
    }

    const data = await response.json();

    if (
        !data ||
        typeof data.session_id !== "string" ||
        !data.session_id.trim()
    ) {
        throw new Error(
            "Invalid conversation response: session_id missing"
        );
    }

    sessionId = data.session_id;

    sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);

    updateSendButton();

    return sessionId;
}

async function initializeConversation() {
    try {
        const storedSession = sessionStorage.getItem(
            SESSION_STORAGE_KEY
        );

        if (storedSession && storedSession.trim()) {
            sessionId = storedSession;
        } else {
            await createConversation();
        }

        updateSendButton();

    } catch (error) {
        console.error(
            "Conversation initialization failed:",
            error
        );

        sessionId = null;

        setStatus("Unable to start conversation", false);

        updateSendButton();
    }
}

// ==================================================
// 8. REQUIREMENTS RENDERING
// ==================================================

function clearRequirements() {
    if (requirementsList) {
        requirementsList.replaceChildren();
    }

    if (requirementsPanel) {
        requirementsPanel.hidden = true;
    }
}

function renderRequirements(requirements) {
    clearRequirements();

    if (
        !requirements ||
        typeof requirements !== "object" ||
        Array.isArray(requirements) ||
        !requirementsList ||
        !requirementsPanel
    ) {
        return;
    }

    const labels = {
        location: "Preferred Location",
        property_type: "Property Type",
        configuration: "Configuration",
        possession_before: "Possession Before",
        availability: "Availability"
    };

    const entries = [];

    // Combine available budget boundaries.
    const min = requirements.budget_min;
    const max = requirements.budget_max;

    if (hasValue(min) || hasValue(max)) {
        let budgetText;

        if (hasValue(min) && hasValue(max)) {
            budgetText =
                `${formatPrice(min)} - ${formatPrice(max)}`;

        } else if (hasValue(min)) {
            budgetText = `From ${formatPrice(min)}`;

        } else {
            budgetText = `Up to ${formatPrice(max)}`;
        }

        entries.push(["Budget", budgetText]);
    }

    // Render remaining supported requirement fields.
    Object.entries(labels).forEach(([key, label]) => {
        const value = requirements[key];

        if (!hasValue(value)) return;

        const displayValue =
            key === "possession_before"
                ? formatDate(value)
                : Array.isArray(value)
                    ? value.join(", ")
                    : String(value);

        entries.push([label, displayValue]);
    });

    if (entries.length === 0) {
        return;
    }

    entries.forEach(([label, value]) => {
        const item = document.createElement("li");

        const labelElement = createTextElement(
            "strong",
            "requirement-label",
            `${label}: `
        );

        const valueElement = createTextElement(
            "span",
            "requirement-value",
            value
        );

        item.append(labelElement, valueElement);

        requirementsList.appendChild(item);
    });

    requirementsPanel.hidden = false;
}

// ==================================================
// 9. PROPERTY CARD HELPERS
// ==================================================

function appendPropertyField(card, label, value) {
    if (!hasValue(value)) return;

    const field = document.createElement("p");

    const labelElement = createTextElement(
        "strong",
        "property-field-label",
        `${label}: `
    );

    const valueElement = createTextElement(
        "span",
        "property-field-value",
        Array.isArray(value) ? value.join(", ") : value
    );

    field.append(labelElement, valueElement);

    card.appendChild(field);
}

function getPropertyPrice(property) {
    const min = property.price_min;
    const max = property.price_max;
    const listing = property.listing_price;

    if (hasValue(min) && hasValue(max)) {
        if (Number(min) === Number(max)) {
            return formatPrice(min);
        }

        return `${formatPrice(min)} - ${formatPrice(max)}`;
    }

    if (hasValue(min)) {
        return `From ${formatPrice(min)}`;
    }

    if (hasValue(max)) {
        return `Up to ${formatPrice(max)}`;
    }

    if (hasValue(listing)) {
        return formatPrice(listing);
    }

    return null;
}

function createPropertyCard(property) {
    const card = document.createElement("article");

    card.className = "property-card";

    // Project name.
    const title = createTextElement(
        "h4",
        "property-title",
        hasValue(property.project_name)
            ? property.project_name
            : "Property Listing"
    );

    card.appendChild(title);

    // Property data provenance.
    // A non-synthetic flag alone does not establish that a record
    // is live or verified.
    let provenanceLabel = "Source / Verification Unknown";

    if (property.is_synthetic === true) {
        provenanceLabel = "Synthetic Demo Record";

    } else if (
        property.is_synthetic === false &&
        hasValue(property.source_name)
    ) {
        provenanceLabel = "Dataset Record — Not Live-Verified";
    }

    const provenanceBadge = createTextElement(
        "span",
        "provenance-badge",
        provenanceLabel
    );

    card.appendChild(provenanceBadge);

    // Basic property information.
    appendPropertyField(card, "Location", property.location);
    appendPropertyField(card, "Property Type", property.property_type);
    appendPropertyField(card, "Configuration", property.configuration);

    // Price.
    const price = getPropertyPrice(property);

    if (hasValue(price)) {
        appendPropertyField(card, "Price", price);
    }

    // Additional details.
    appendPropertyField(
        card,
        "Possession Date",
        formatDate(property.possession_date)
    );

    appendPropertyField(
        card,
        "Availability",
        property.availability_status
    );

    appendPropertyField(
        card,
        "Amenities",
        property.amenities
    );

    appendPropertyField(
        card,
        "Nearby Locations",
        property.nearby_locations
    );

    appendPropertyField(
        card,
        "Description",
        property.description
    );

    appendPropertyField(
        card,
        "Price Note",
        property.price_note
    );

    appendPropertyField(
        card,
        "Source",
        property.source_name
    );

    return card;
}

// ==================================================
// 10. CALLBACK STATUS RENDERING
// ==================================================

function createCallbackCard(data) {
    const card = document.createElement("div");
    card.className = "callback-card";

    const intent = data.intent;
    const nextAction = data.next_action || {};
    const toolResults = data._tool_results || [];

    // Determine callback state from intent and next_action.
    if (intent === "callback_request" && nextAction.action === "await_user") {
        // Callback was successfully created.
        card.classList.add("callback-success");

        const icon = createTextElement("span", "callback-icon", "✅");
        const heading = createTextElement("h4", "callback-heading", "Callback Scheduled");
        const detail = createTextElement(
            "p",
            "callback-detail",
            "Your callback request has been submitted. Our team will reach out to you at the preferred date and time."
        );

        card.append(icon, heading, detail);
        return card;
    }

    if (intent === "callback_request" && nextAction.action === "ask_clarification") {
        // Callback flow in progress — needs more info or consent.
        card.classList.add("callback-pending");

        const icon = createTextElement("span", "callback-icon", "\u{1F4DE}");
        const heading = createTextElement("h4", "callback-heading", "Callback Request");
        const detail = createTextElement(
            "p",
            "callback-detail",
            "Please provide the requested information to complete your callback request."
        );

        card.append(icon, heading, detail);
        return card;
    }

    // No callback-specific card needed.
    return null;
}

function renderCallbackStatus(data) {
    if (!chatContainer || !data) return;

    const card = createCallbackCard(data);
    if (!card) return;

    chatContainer.appendChild(card);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}


// ==================================================
// 11. PROPERTY LIST RENDERING
// ==================================================

function clearProperties() {
    if (propertiesList) {
        propertiesList.replaceChildren();
    }

    if (propertiesPanel) {
        propertiesPanel.hidden = true;
    }
}

function renderProperties(properties) {
    clearProperties();

    if (
        !Array.isArray(properties) ||
        properties.length === 0 ||
        !propertiesList ||
        !propertiesPanel
    ) {
        return;
    }

    properties.forEach((property) => {
        if (
            !property ||
            typeof property !== "object" ||
            Array.isArray(property)
        ) {
            return;
        }

        const card = createPropertyCard(property);

        propertiesList.appendChild(card);
    });

    if (propertiesList.childElementCount > 0) {
        propertiesPanel.hidden = false;
    }
}

// ==================================================
// 12. AGENT API REQUEST
// ==================================================

async function requestAgentResponse(message) {
    return fetch("/api/agent/turn", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            session_id: sessionId,
            message: message
        })
    });
}

// ==================================================
// 13. MESSAGE SENDING
// ==================================================

async function sendMessage(message) {
    const cleanMessage = message.trim();

    if (!cleanMessage || busy) return;

    busy = true;

    updateSendButton();

    if (msgInput) {
        msgInput.disabled = true;
    }

    // Display the user message once.
    renderMessage(cleanMessage, "user");

    // Show typing indicator while waiting.
    showTypingIndicator();

    try {
        if (!sessionId) {
            await createConversation();
        }

        // Initial request.
        let response = await requestAgentResponse(cleanMessage);

        // Recover from an expired or invalid session.
        if (response.status === 404) {
            console.warn(
                "Session expired. Creating a new conversation."
            );

            sessionStorage.removeItem(SESSION_STORAGE_KEY);

            sessionId = null;

            await createConversation();

            // Retry the original message exactly once.
            response = await requestAgentResponse(cleanMessage);
        }

        if (!response.ok) {
            let errorDetail = `HTTP ${response.status}`;

            try {
                const errorData = await response.json();

                if (errorData.detail) {
                    errorDetail = errorData.detail;
                }

            } catch {
                // Retain the HTTP status if the body is not JSON.
            }

            throw new Error(errorDetail);
        }

        const data = await response.json();

        if (!data || typeof data.reply !== "string") {
            throw new Error(
                "Invalid response received from the agent."
            );
        }

        // Remove typing indicator before showing the reply.
        hideTypingIndicator();

        // Assistant reply.
        renderMessage(data.reply, "assistant");

        // Callback status card (if applicable).
        renderCallbackStatus(data);

        // Structured customer requirements.
        renderRequirements(data.requirements);

        // Matched property cards.
        renderProperties(data.matched_properties);

        // Clear input only after successful completion.
        if (msgInput) {
            msgInput.value = "";
            autoResizeInput();
        }

        setStatus("Connected", true);

    } catch (error) {
        console.error("Message sending failed:", error);

        hideTypingIndicator();

        // Preserve the original message for retry.
        if (msgInput) {
            msgInput.value = cleanMessage;
        }

        showError(
            "I couldn't process your message. Please try again."
        );

        showToast(
            error.message || "Something went wrong. Please try again.",
            "error"
        );

        setStatus("Request failed", false);

    } finally {
        busy = false;

        if (msgInput) {
            msgInput.disabled = false;
        }

        updateSendButton();

        if (msgInput) {
            msgInput.focus();
        }
    }
}

// ==================================================
// 14. EVENT HANDLERS
// ==================================================

function handleSendClick() {
    if (!msgInput) return;

    const message = msgInput.value.trim();

    if (message) {
        sendMessage(message);
    }
}

function handleInputKeydown(event) {
    if (
        event.key === "Enter" &&
        !event.shiftKey &&
        !event.isComposing
    ) {
        event.preventDefault();

        handleSendClick();
    }
}

function initializeSuggestionButtons() {
    document.querySelectorAll(".suggest-btn").forEach((button) => {
        button.addEventListener("click", () => {
            if (busy || !msgInput) return;

            const prompt = button.textContent.trim();

            if (!prompt) return;

            msgInput.value = prompt;

            updateSendButton();

            sendMessage(prompt);
        });
    });
}

// ==================================================
// 15. APPLICATION INITIALIZATION
// ==================================================

async function init() {
    updateSendButton();

    // Send button.
    if (sendBtn) {
        sendBtn.addEventListener("click", handleSendClick);
    }

    // Text input.
    if (msgInput) {
        msgInput.addEventListener("input", () => {
            updateSendButton();
            autoResizeInput();
        });

        msgInput.addEventListener(
            "keydown",
            handleInputKeydown
        );
    }

    // Suggested prompts.
    initializeSuggestionButtons();

    // Backend health.
    await checkBackend();

    // Restore or create conversation.
    await initializeConversation();

    updateSendButton();
}

// ==================================================
// 16. DOM READY
// ==================================================

window.addEventListener("DOMContentLoaded", init);
// Export selected functions for Node.js tests.
// This does not affect normal browser execution.
if (typeof module !== "undefined" && module.exports) {
    module.exports = {
        parseMarkdown,
        splitTableRow,
        createPropertyCard,
        createCallbackCard,
    };
}