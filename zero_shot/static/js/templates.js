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
};
