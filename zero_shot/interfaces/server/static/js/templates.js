// Example question sets loaded by the "Load example" dropdown. All use the same
// state so the question types can be compared directly.

export const templates = {
  choice: {
    question: {
      department: {
        type: "choice",
        instructions: "Which team should handle this?",
        criteria: {
          billing: "Payments, invoicing, refunds",
          technical: "Bugs, outages, integrations",
          sales: "Pricing, upgrades, new accounts",
        },
      },
    },
    state: { input: "Help! My payouts have been failing for 3 days." },
  },
  noul: {
    question: {
      is_urgent: {
        type: "noul",
        instructions: "Does this convey urgency?",
        criteria: { true: "Explicitly time-sensitive", false: "No urgency expressed" },
      },
    },
    state: { input: "Help! My payouts have been failing for 3 days." },
  },
  score: {
    question: {
      frustration: {
        type: "score",
        instructions: "How frustrated is the customer?",
        criteria: ["Calm", "Frustrated", "Very angry"],
      },
    },
    state: { input: "Help! My payouts have been failing for 3 days." },
  },
  mixed: {
    question: {
      is_urgent: {
        type: "noul",
        instructions: "Does this convey urgency?",
        criteria: { true: "Explicitly time-sensitive", false: "No urgency expressed" },
      },
      frustration: {
        type: "score",
        instructions: "How frustrated is the customer?",
        criteria: ["Calm", "Frustrated", "Very angry"],
      },
      department: {
        type: "choice",
        instructions: "Which team should handle this?",
        criteria: {
          billing: "Payments, invoicing, refunds",
          technical: "Bugs, outages, integrations",
          sales: "Pricing, upgrades, new accounts",
        },
      },
    },
    state: { input: "Help! My payouts have been failing for 3 days." },
  },
  image: {
    mode: "image",
    question: {
      screen: {
        type: "choice",
        instructions: "Which kind of screen is this?",
        criteria: {
          login: "A sign-in form",
          checkout: "A payment or checkout page",
          error: "An error or failure page",
          dashboard: "An app or dashboard view",
        },
      },
      has_close: {
        type: "noul",
        instructions: "Is a close (X) or dismiss control visible?",
        criteria: {
          true: "A close or dismiss control is visible",
          false: "No close or dismiss control",
        },
      },
      clutter: {
        type: "score",
        instructions: "How cluttered is the screen?",
        criteria: ["Minimal", "Clean", "Busy", "Overwhelming"],
      },
    },
  },
};
