// Display metadata complements the original Markdown; it does not replace it.
// Highlights below are grounded in the existing project write-ups.
export type ProjectDisplay = {
  title: string;
  category: "LLMs" | "Reinforcement Learning" | "Statistical Learning";
  summary: string;
  highlight: string;
  tags: string[];
  status?: string;
};

export const projectDisplay: Record<string, ProjectDisplay> = {
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
    title: "Distributed reinforcement learning",
    category: "Reinforcement Learning",
    summary:
      "Building single-core and distributed DQN implementations with separate collectors, model, replay memory, and evaluation.",
    highlight: "From a single DQN to distributed actors",
    tags: ["DQN", "Distributed Training", "CartPole"],
  },
  "rl-fine-tunning": {
    title: "LLM alignment with GRPO",
    category: "LLMs",
    summary:
      "A personal exploration of Group Relative Policy Optimization for reinforcement learning based LLM fine-tuning.",
    highlight: "Exploring reinforcement learning for alignment",
    tags: ["GRPO", "LLM Alignment", "Reinforcement Learning"],
    status: "In progress",
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
  "database-join",
  "vul-detection",
  "distributed-agents",
  "image-denoising",
];
