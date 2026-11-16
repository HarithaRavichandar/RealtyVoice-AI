const { test, describe } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

// --------------------------------------------------
// Minimal DOM implementation for dependency-free tests
// --------------------------------------------------

class TestNode {
    constructor(type, tagName = null, text = "") {
        this.type = type;
        this.tagName = tagName;
        this.text = text;
        this.childNodes = [];
        this.attributes = {};
    }

    appendChild(node) {
        if (node.type === "fragment") {
            this.childNodes.push(...node.childNodes);
            node.childNodes = [];
        } else {
            this.childNodes.push(node);
        }

        return node;
    }

    append(...nodes) {
        nodes.forEach(node => this.appendChild(node));
    }

    setAttribute(name, value) {
        this.attributes[name] = String(value);
    }

    get textContent() {
        if (this.type === "text") {
            return this.text;
        }

        return this.childNodes
            .map(child => child.textContent)
            .join("");
    }

    set textContent(value) {
        this.childNodes = [];
        this.text = String(value);

        if (this.type !== "text" && this.text.length > 0) {
            this.childNodes.push(
                new TestNode("text", null, this.text)
            );
        }
    }
}

// --------------------------------------------------
// Mock document
// --------------------------------------------------

const documentMock = {
    createElement(tag) {
        return new TestNode("element", tag);
    },

    createTextNode(text) {
        return new TestNode("text", null, String(text));
    },

    createDocumentFragment() {
        return new TestNode("fragment");
    },

    getElementById() {
        return null;
    },

    querySelectorAll() {
        return [];
    }
};

// --------------------------------------------------
// Load actual parser from app.js
// --------------------------------------------------

const appPath = path.join(
    __dirname,
    "..",
    "js",
    "app.js"
);

const source = fs.readFileSync(appPath, "utf8");

const context = {
    document: documentMock,
    window: {
        addEventListener() { }
    },
    console,
    Intl,
    Date,
    String,
    Number,
    Array,
    Object,
    RegExp
};

vm.createContext(context);

// Expose the actual functions only inside the test VM.
vm.runInContext(
    source + `
        globalThis.testExports = {
            parseMarkdown,
            splitTableRow,
            createPropertyCard
        };
    `,
    context
);

const {
    parseMarkdown,
    splitTableRow,
    createPropertyCard
} = context.testExports;

// --------------------------------------------------
// DOM inspection helpers
// --------------------------------------------------

function findElements(node, tagName) {
    const results = [];

    for (const child of node.childNodes || []) {
        if (child.tagName === tagName) {
            results.push(child);
        }

        results.push(
            ...findElements(child, tagName)
        );
    }

    return results;
}

// --------------------------------------------------
// Markdown rendering tests
// --------------------------------------------------

describe("RealtyVoice AI Markdown rendering", () => {

    test("renders bold text", () => {
        const result = parseMarkdown(
            "This is **bold** text."
        );

        const strong = findElements(result, "strong");

        assert.equal(strong.length, 1);
        assert.equal(strong[0].textContent, "bold");
    });

    test("renders italic text", () => {
        const result = parseMarkdown(
            "This is *italic* text."
        );

        const italic = findElements(result, "em");

        assert.equal(italic.length, 1);
        assert.equal(italic[0].textContent, "italic");
    });

    test("renders unordered lists", () => {
        const result = parseMarkdown(
            "- Chennai\n- Bangalore\n- Hyderabad"
        );

        const lists = findElements(result, "ul");
        const items = findElements(result, "li");

        assert.equal(lists.length, 1);
        assert.equal(items.length, 3);
        assert.equal(items[0].textContent, "Chennai");
    });

    test("renders ordered lists", () => {
        const result = parseMarkdown(
            "1. First\n2. Second\n3. Third"
        );

        const lists = findElements(result, "ol");
        const items = findElements(result, "li");

        assert.equal(lists.length, 1);
        assert.equal(items.length, 3);
    });

    test("renders Markdown tables", () => {
        const result = parseMarkdown(
            "| Property | Location |\n" +
            "|---|---|\n" +
            "| Residency | Chennai |\n" +
            "| Apartments | Bangalore |"
        );

        const tables = findElements(result, "table");
        const headers = findElements(result, "th");
        const cells = findElements(result, "td");

        assert.equal(tables.length, 1);
        assert.equal(headers.length, 2);
        assert.equal(cells.length, 4);
        assert.equal(cells[0].textContent, "Residency");
    });

    test("renders headings", () => {
        const result = parseMarkdown(
            "## Available Properties"
        );

        const headings = findElements(result, "h2");

        assert.equal(headings.length, 1);
        assert.equal(
            headings[0].textContent,
            "Available Properties"
        );
    });

    test("renders inline code", () => {
        const result = parseMarkdown(
            "Use `property_id` here."
        );

        const code = findElements(result, "code");

        assert.equal(code.length, 1);
        assert.equal(code[0].textContent, "property_id");
    });

    test("preserves line breaks", () => {
        const result = parseMarkdown(
            "First line\nSecond line"
        );

        const breaks = findElements(result, "br");

        assert.equal(breaks.length, 1);
    });

});

// --------------------------------------------------
// Markdown security tests
// --------------------------------------------------

describe("Markdown security", () => {

    test("does not create executable script elements", () => {
        const result = parseMarkdown(
            '<script>alert("XSS")</script>'
        );

        assert.equal(
            findElements(result, "script").length,
            0
        );

        assert.equal(
            result.textContent,
            '<script>alert("XSS")</script>'
        );
    });

    test("does not interpret raw HTML as DOM elements", () => {
        const result = parseMarkdown(
            '<img src=x onerror="alert(1)">'
        );

        assert.equal(
            findElements(result, "img").length,
            0
        );

        assert.equal(
            result.textContent,
            '<img src=x onerror="alert(1)">'
        );
    });

    test("rejects javascript URL links", () => {
        const result = parseMarkdown(
            "[Click](javascript:alert(1))"
        );

        assert.equal(
            findElements(result, "a").length,
            0
        );
    });

    test("allows HTTPS links safely", () => {
        const result = parseMarkdown(
            "[Visit](https://example.com)"
        );

        const links = findElements(result, "a");

        assert.equal(links.length, 1);

        assert.equal(
            links[0].attributes.href,
            "https://example.com"
        );

        assert.equal(
            links[0].attributes.rel,
            "noopener noreferrer"
        );
    });

});

// --------------------------------------------------
// Markdown table parsing tests
// --------------------------------------------------

test("splits Markdown table cells correctly", () => {
    const result = splitTableRow(
        "| Name | Location |"
    );

    assert.deepEqual(
        Array.from(result),
        [
            "Name",
            "Location"
        ]
    );
});
// --------------------------------------------------
// Property provenance and price-note tests
// --------------------------------------------------

describe("Property card provenance", () => {

    test("labels synthetic records correctly", () => {
        const card = createPropertyCard({
            property_id: 101,
            project_name: "Demo Apartment",
            location: "Delhi",
            property_type: "apartment",
            is_synthetic: true,
            source_name: "synthetic_demo"
        });

        assert.ok(
            card.textContent.includes("Synthetic Demo Record")
        );
    });

    test("labels records with missing provenance as unknown", () => {
        const card = createPropertyCard({
            property_id: 102,
            project_name: "Unknown Apartment",
            location: "Chennai",
            property_type: "apartment",
            is_synthetic: false,
            source_name: null
        });

        assert.ok(
            card.textContent.includes("Source / Verification Unknown")
        );
    });

    test("displays price_note when provided", () => {
        const card = createPropertyCard({
            property_id: 103,
            project_name: "Test Apartment",
            location: "Bangalore",
            property_type: "apartment",
            is_synthetic: false,
            source_name: "test_dataset",
            price_note: "Indicative price from historical dataset"
        });

        assert.ok(
            card.textContent.includes("Price Note")
        );

        assert.ok(
            card.textContent.includes(
                "Indicative price from historical dataset"
            )
        );
    });

});