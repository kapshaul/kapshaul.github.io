// Display metadata complements the original Markdown; it does not replace it.
// Highlights below are grounded in the existing project write-ups.
export type ProjectDisplay = {
  title: string;
  category: "AI Agents" | "LLMs" | "Reinforcement Learning" | "Statistical Learning";
  summary: string;
  highlight: string;
  tags: string[];
  status?: string;
  period?: string;
  video?: {
    src: string;
    poster: string;
    width: number;
    height: number;
    title: string;
    caption: string;
  };
};

export const projectDisplay: Record<string, ProjectDisplay> = {
  jarvis: {
    title: "JARVIS",
    category: "AI Agents",
    summary:
      "A conversational CAD agent that connects natural-language requests to modeling tools, geometry validation, and an interactive 3D view.",
    highlight: "Natural language → CAD tools → verified geometry",
    tags: ["LangChain", "Tool Calling", "CAD", "3D Visualization"],
    status: "In progress",
    period: "Mar. 2026 – Present",
    video: {
      src: "/projects/jarvis/bracket-ribs-demo.mp4",
      poster: "/projects/jarvis/bracket-ribs-poster.jpg",
      width: 1280,
      height: 596,
      title: "JARVIS demo: selecting a bracket face and adding two reinforcing ribs",
      caption:
        "Select a bracket face, request two reinforcing ribs, and inspect the updated geometry. This 10-second local demo uses a scripted command with real CAD tool execution.",
    },
  },
  "construction-ai-agent": {
    title: "Civil & Construction AI Agent",
    category: "AI Agents",
    summary:
      "Developing domain agents for workflow automation, with multi-agent orchestration, tool integration, and memory and context management.",
    highlight: "Domain knowledge connected to agent execution",
    tags: ["LangChain", "OpenAI Agents SDK", "Multi-Agent Orchestration"],
    status: "In progress",
    period: "Jun. 2025 – Present",
  },
  "database-join": {
    title: "Learning to optimize database joins",
    category: "Reinforcement Learning",
    summary:
      "Teaching PostgreSQL join operators to learn during query execution and share information across operators.",
    highlight: "Online learning inside the query engine",
    tags: ["Online Learning", "PostgreSQL", "Query Optimization"],
  },
  "vul-detection": {
    title: "Efficient LLMs for vulnerability detection",
    category: "LLMs",
    summary:
      "Adapting code language models with 4-bit quantization and LoRA, exploring dataset, context length, and loss choices.",
    highlight: "13B model · 4-bit QLoRA adaptation",
    tags: ["QLoRA", "LLM Fine-Tuning", "Code Security"],
  },
  "distributed-agents": {
    title: "DQN with parallel experience collection",
    category: "Reinforcement Learning",
    summary:
      "Implementing and comparing DQN with sequential and parallel experience collection, using a shared learner, replay memory, and separate evaluation.",
    highlight: "Parallel collectors feeding a shared DQN learner",
    tags: ["DQN", "Parallel Experience Collection", "CartPole"],
  },
  "image-denoising": {
    title: "Recovering images from Poisson noise",
    category: "Statistical Learning",
    summary:
      "Deriving and simulating expectation maximization for CT image denoising, with estimation analysis and CRLB evaluation.",
    highlight: "Statistical estimation for CT reconstruction",
    tags: ["EM Algorithm", "Image Denoising", "MLE"],
  },
  "kf-robot": {
    title: "Sensor fusion for robot orientation",
    category: "Statistical Learning",
    summary:
      "Fusing accelerometer and gyroscope measurements with an extended Kalman filter for a walking robot’s attitude estimation.",
    highlight: "Accelerometer + gyroscope → orientation",
    tags: ["Extended Kalman Filter", "Sensor Fusion", "Robotics"],
  },
};

export const featuredSlugs = [
  "jarvis",
  "construction-ai-agent",
  "database-join",
  "vul-detection",
  "distributed-agents",
  "image-denoising",
];
