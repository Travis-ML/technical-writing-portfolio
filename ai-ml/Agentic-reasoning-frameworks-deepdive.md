# Agentic Reasoning Frameworks: A Deep Dive into ReAct, ReWOO, and the Modern AI Agent Landscape

*A comprehensive technical guide to the architectural patterns powering production AI agents in 2024-2025*

**Author:** Travis Lelle ([travis@travisml.ai](mailto:travis@travisml.ai))  

---

## Introduction: Why Different Reasoning Patterns Matter

Traditional LLM interactions are fundamentally reactive, you ask, they respond. But complex tasks require **planning, tool use, reflection, and recovery from failures**. The question facing AI engineers today is: how do you orchestrate these capabilities effectively?

The answer depends on your task's uncertainty profile:
- Can you plan everything upfront? (Low uncertainty)
- Do you need to adapt after every observation? (High uncertainty)  
- Can you afford to explore multiple paths? (Computational budget)

This article dissects the leading agentic design patterns: ReAct, ReWOO, Reflexion, Tree of Thoughts, and CodeAct, offering real-world tradeoff analyses on latency, cost, and complexity. By understanding how to compose these patterns into resilient hybrid systems, you'll gain the concrete knowledge necessary to move your AI projects from fragile proof-of-concepts to robust, scalable production solutions.

---

## 1. ReAct: The Adaptive Workhorse

### Architecture Overview

ReAct (Reasoning + Acting) operates in a **tight thought-action-observation loop** (1). The paradigm was introduced by researchers at Princeton University and Google Research in 2022 and has become one of the foundational patterns for interactive agents.

```
1. Thought: "I need to find the 2024 Australian Open winner"
2. Action: Google["2024 Australian Open winner"]
3. Observation: "Jannik Sinner defeated Daniil Medvedev"
4. Thought: "Now I need his hometown"
5. Action: Google["Jannik Sinner hometown"]
6. Observation: "San Candido, Italy"
7. Thought: "I have enough information"
8. Answer: "Jannik Sinner from San Candido, Italy"
```

**Documentation & Resources:**
- Original Paper: https://arxiv.org/abs/2210.03629
- Official Implementation: https://github.com/ysymyth/ReAct
- LangChain ReAct Agent: https://python.langchain.com/docs/modules/agents/agent_types/react

### Why It Works

Each step informs the next. The agent can **course-correct in real-time** based on what it actually observes. If a search returns nothing, it immediately adapts. This makes ReAct incredibly robust for exploratory tasks where the path forward isn't clear from the start (1).

### The Token Economics Problem

Here's the critical limitation: every loop iteration includes **the entire conversation history** in the prompt. For a 5-step task:
- Step 1: System prompt + query (~500 tokens)
- Step 2: System prompt + query + step 1 (~1,000 tokens)
- Step 3: System prompt + query + steps 1-2 (~1,500 tokens)
- Step 4: System prompt + query + steps 1-3 (~2,000 tokens)
- Step 5: System prompt + query + steps 1-4 (~2,500 tokens)

**Total**: ~7,500 tokens for what might be answerable in 2,000 tokens with better planning.

### Performance Benchmarks

On the HotpotQA benchmark, ReAct achieves 40.8% accuracy using approximately 10,000 tokens per query (2). While expensive at scale, this adaptability is unbeatable when correctness requires dynamic exploration.

### Production Use Cases & Implementations

**When to use ReAct:**
- **Code debugging** (highly exploratory, unpredictable errors)
- **Customer support** (conversations branch based on user responses)
- **Research assistants** (follow-up questions emerge from initial findings)

**Well-known implementations:**
- **LangChain's ReAct Agent**: The default agent type in LangChain, used by thousands of production applications
- **AutoGPT**: Early autonomous agent that popularized the ReAct pattern for multi-step tasks
- **Anthropic's Claude for Code**: Uses ReAct-style reasoning for iterative code generation and debugging
- **OpenAI's GPT-4 with function calling**: Implements a ReAct-like loop for tool use

---

## 2. ReWOO: The Efficiency Play

### Architecture Overview

ReWOO (Reasoning WithOut Observation) addresses ReAct's token inefficiency through a fundamentally different approach: **plan first, execute later** (2). Introduced by researchers from Microsoft and Virginia Tech in 2023, ReWOO separates planning from execution through three distinct modules.

The architecture uses a **Planner-Worker-Solver** pattern:
1. **Planner**: Creates a complete solution blueprint upfront
2. **Worker**: Executes tool calls based on the blueprint, storing results as evidence
3. **Solver**: Synthesizes the plan and evidence into a final answer

**Documentation & Resources:**
- Original Paper: https://arxiv.org/abs/2305.18323
- Official Implementation: https://github.com/billxbf/ReWOO
- LangGraph ReWOO Tutorial: https://langchain-ai.github.io/langgraph/tutorials/rewoo/rewoo/
- IBM ReWOO Guide: https://www.ibm.com/think/topics/rewoo

### The Variable Substitution Magic

The key innovation is **placeholder-based planning**:

```python
# Planner generates this UPFRONT (one LLM call):
Plan: Search for 2024 Australian Open winner
#E1 = Google["2024 Australian Open men's winner"]

Plan: Find the winner's hometown  
#E2 = Google["hometown of #E1"]  # Uses placeholder!

Plan: Synthesize the answer
#E3 = LLM["What is the hometown of #E1 given #E2"]
```

Workers execute these steps sequentially, substituting variables as they go. The Solver never sees the full conversation history, just the variables and results (2).

### Performance Numbers

ReWOO achieves 42.4% accuracy on HotpotQA using only 2,000 tokens—comparable accuracy to ReAct but with **80% fewer tokens** (2). On multi-step reasoning benchmarks, ReWOO demonstrates 5x token efficiency while maintaining a 4% accuracy improvement over baseline methods (2).

### The Critical Tradeoff

ReWOO commits to a plan before seeing any results. If step 1 returns garbage, it can't adapt—it'll blindly execute steps 2 and 3 anyway. You can add replanning logic, but now you're building a hybrid system.

As IBM's research notes, "ReWOO would not be optimal for debugging Python code, an exploratory and iterative process where each fix might yield new errors, making pre-laid plans quickly obsolete" (2).

### Production Use Cases & Implementations

**When to use ReWOO:**
- **Multi-hop questions** with predictable structure ("Who won X? Where are they from?")
- **Token-constrained environments** (massive scale, cost-sensitive applications)
- **Tasks with reliable tools** (if your search API is flaky, ReWOO will fail spectacularly)
- **Batch processing workflows** where efficiency matters more than adaptability

**Well-known implementations:**
- **IBM Watson Assistant**: Integrates ReWOO for cost-efficient multi-step reasoning
- **LangGraph's ReWOO module**: Production-ready implementation in the LangGraph framework
- **Microsoft Semantic Kernel**: Supports ReWOO-style planning for enterprise agents
- **Process mining applications**: Used for inserting domain knowledge into process discovery workflows (3)

---

## 3. Reflexion: Self-Improving Agents Through Verbal Reinforcement

### The Core Innovation

Reflexion takes a fundamentally different approach from both ReAct and ReWOO. Instead of focusing purely on planning or execution patterns, Reflexion introduces **verbal reinforcement learning** - the ability for agents to learn from their mistakes across multiple episodes (4).

Reflexion converts feedback from the environment into linguistic self-reflection, which becomes context for the LLM agent in subsequent episodes, enabling rapid learning from prior mistakes (4).

**Documentation & Resources:**
- Original Paper: https://arxiv.org/abs/2303.11366
- LangGraph Reflexion Tutorial: https://langchain-ai.github.io/langgraph/tutorials/reflection/reflection/
- Prompt Engineering Guide: https://www.promptingguide.ai/techniques/reflexion

### Architecture Components

The Reflexion framework consists of three key modules (4):

1. **Actor**: Executes tasks (uses ReAct or CoT internally)
2. **Evaluator**: Scores outcomes using reward signals
3. **Self-Reflection**: Generates verbal feedback about failures
4. **Memory**: Stores reflections for future episodes

### Example in Action

```
Episode 1:
Task: "Find restaurants within 1km of San Francisco"
Attempt: Searches "San Francisco restaurants" (too broad)
Result: FAIL
Reflection: "I didn't specify the distance constraint in my search. 
Next time, include '1km' or 'nearby' in the query."

Episode 2:
Task: Same task
Memory: [Previous reflection]
Attempt: Searches "restaurants within 1km San Francisco"  
Result: SUCCESS
```

### Performance Benchmarks

On sequential decision-making tasks (AlfWorld), ReAct + Reflexion significantly outperforms standalone ReAct by completing 130/134 tasks versus ReAct's lower success rate (4). The paper demonstrates an 8% absolute improvement over episodic memory alone, supporting the argument that verbal self-reflection enables more effective learning than simple trajectory replay.

### Production Use Cases & Implementations

**When to use Reflexion:**
- **Iterative code generation** (fix bugs based on test failures)
- **Interactive environments** (game-playing agents that learn strategies)
- **Long-running optimization tasks** where multiple attempts are acceptable
- **Quality improvement loops** in content generation

**Well-known implementations:**
- **OpenAI's Code Interpreter improvements**: Incorporates reflection-like patterns for iterative debugging
- **LangGraph Reflection Agents**: Built-in support for self-critique and improvement cycles
- **CodeTree**: Uses reflection for tree-based exploration in code generation (5)
- **Research agent systems**: Academic paper analysis tools that improve through self-critique

**The Critical Limitation:**

Reflexion requires **multiple episodes** (attempts). If you need a one-shot answer, it doesn't help. It's designed for systems that can retry and improve over time, not for immediate responses.

---

## 4. Tree of Thoughts (ToT): Deliberate Problem Solving Through Exploration

### The Paradigm Shift

Instead of committing to a single reasoning path, Tree of Thoughts enables **exploration of multiple branches** and selection of the best solution (6). Introduced by researchers at Princeton University and Google DeepMind in 2023, ToT represents a fundamental departure from linear reasoning patterns.

**Documentation & Resources:**
- Original Paper: https://arxiv.org/abs/2305.10601
- Official Implementation: https://github.com/princeton-nlp/tree-of-thought-llm
- LangGraph ToT Tutorial: https://langchain-ai.github.io/langgraph/tutorials/tot/tot/
- Prompt Engineering Guide: https://www.promptingguide.ai/techniques/tot

### How ToT Works

ToT maintains a tree of thoughts, where thoughts represent coherent language sequences serving as intermediate steps toward problem solving (6). The framework enables LLMs to perform deliberate decision-making by:

1. **Generating multiple reasoning paths** at each decision point
2. **Self-evaluating** the quality of each path
3. **Looking ahead** to anticipate outcomes
4. **Backtracking** when a path proves unfruitful

Think of it as **breadth-first search over reasoning paths**:

```
Root: "Solve 3 + 5 * 2"
├─ Branch A: "First add 3+5=8, then multiply 8*2=16" (WRONG)
├─ Branch B: "First multiply 5*2=10, then add 3+10=13" (CORRECT)
└─ Branch C: "Multiply everything: 3*5*2=30" (WRONG)

Evaluator scores each branch, selects Branch B.
```

### Performance Benchmarks

The improvement over traditional approaches is dramatic. On the Game of 24 benchmark, GPT-4 with standard chain-of-thought prompting solved only 4% of tasks. With Tree of Thoughts, the success rate jumped to 74% (6).

### Language Agent Tree Search (LATS): The Advanced Variant

LATS extends ToT by combining reflection, evaluation, and Monte Carlo Tree Search (MCTS) to achieve better overall task performance compared to ReAct, Reflexion, or standard ToT (7).

LATS adds:
- **Selection**: Pick the most promising node using MCTS principles
- **Expansion**: Generate N actions from that node  
- **Simulation**: Execute them in parallel
- **Backpropagation**: Update scores based on outcomes

**Documentation:**
- LATS Paper: https://arxiv.org/abs/2310.04406
- LangGraph LATS Tutorial: https://langchain-ai.github.io/langgraph/tutorials/lats/lats/

### Cost Implications

ToT and LATS are **computationally expensive**. If ToT explores 5 branches at 3 levels deep, that's 5³ = 125 LLM calls. But for critical reasoning tasks (medical diagnosis, legal analysis, strategic planning), the accuracy boost often justifies the cost.

### Production Use Cases & Implementations

**When to use ToT/LATS:**
- **High-stakes decisions** where correctness >> cost
- **Complex reasoning** (math proofs, strategic planning, scientific hypothesis generation)
- **Tasks with verifiable outputs** (you can score each branch objectively)
- **Creative tasks** requiring exploration of multiple approaches

**Well-known implementations:**
- **Claude's extended thinking mode**: Uses ToT-like exploration for complex reasoning
- **o1 model series from OpenAI**: Reportedly uses tree search for enhanced reasoning
- **AlphaCode-style systems**: Apply tree search to code generation
- **Math reasoning systems**: Academic benchmarks like GSM8K benefit from ToT approaches

---

## 5. CodeAct / Tree-of-Code: Code as the Universal Action Space

### The Revolutionary Concept

Instead of calling tools via JSON or structured commands, CodeAct treats **Python code itself as the action** (8). This paradigm shift, introduced in 2024, fundamentally expands what agents can accomplish.

**Documentation & Resources:**
- CodeAct concepts in LangGraph: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/
- Tree-of-Code Paper: https://arxiv.org/abs/2412.15305

### The CodeAct Pattern

When facing a request like "Calculate the 95th percentile of sales data and visualize the distribution," the agent generates code in a self-correcting loop: Reason, Code, Execute, Observe/Debug (8).

```python
# Agent generates:
import pandas as pd
import matplotlib.pyplot as plt

data = fetch_sales_data()  # Tool call embedded in code
p95 = data['sales'].quantile(0.95)
data['sales'].hist()
plt.axvline(p95, color='red', label='95th percentile')
plt.savefig('output.png')
```

### Tree-of-Code: CodeAct Meets Exploration

Tree-of-Code (ToC) combines CodeAct with tree-based exploration (9). Instead of generating code sequentially, ToC:

1. **Generates multiple complete solutions** as tree nodes
2. **Executes them in parallel** at each tree layer
3. **Collects successful executions** for voting
4. **Selects the best solution** based on execution success

Recent benchmarks show ToC outperforms sequential CodeAct on complex multi-turn, multi-tool tasks (9).

### Why Code > Traditional Tool Calls

**Richer action space**: Can perform arbitrary computation, not just predefined tools
**Self-verification**: Code either runs successfully or throws an error (binary feedback)
**Composability**: Chain operations without framework-specific glue code
**Debugging potential**: Failed code provides stack traces and error messages

### Security Considerations

Code execution is **inherently dangerous**. Production systems require:
- **Sandboxing**: Docker containers or restricted execution environments
- **Resource limits**: CPU, memory, and time constraints
- **Code review**: Static analysis before execution for enterprise deployments

### Production Use Cases & Implementations

**When to use CodeAct:**
- **Data analysis tasks** requiring computation, not just information retrieval
- **Scientific computing** where calculations are core to the task
- **Automation workflows** that manipulate files, data, or systems
- **Tasks with verifiable outputs** (tests, assertions, expected results)

**Well-known implementations:**
- **OpenAI's Code Interpreter**: Pioneered code-as-action for ChatGPT
- **Anthropic's Claude Code**: Full-featured coding agent using code execution
- **E2B Code Interpreter**: Open-source sandboxed code execution for agents
- **Jupyter AI**: Integrates code execution into notebook environments

---

## Production Framework Landscape (2024-2025)

The theoretical patterns above are implemented in practice through several competing frameworks. Here's what's actually being used in production systems.

### LangGraph: The State Machine Champion

LangGraph offers a graph-based runtime perfect for stateful workflows with fine-grained flow management (10). Built by the LangChain team, it introduces a powerful way to structure agents as stateful graphs rather than linear chains.

**Documentation & Resources:**
- Official Documentation: https://langchain-ai.github.io/langgraph/
- GitHub Repository: https://github.com/langchain-ai/langgraph
- Conceptual Guide: https://langchain-ai.github.io/langgraph/concepts/

**Mental model**: You're building a **flowchart**. Each node is an agent/tool, edges define transitions.

**Best for:**
- Financial modeling (strict compliance, audit trails, deterministic paths)
- Healthcare workflows (HIPAA compliance, regulatory requirements)
- Complex RAG pipelines (retrieve → rerank → generate → verify)
- Multi-step processes requiring explicit state management

**The learning curve**: Steeper than alternatives. Requires thinking in graphs and managing state explicitly.

**Production adopters**: Teams using LangChain at scale; enterprises requiring compliance and auditability; systems with complex conditional logic.

### CrewAI: The Team Simulator

CrewAI treats agents like collaborators with roles, goals, and tools (10). Launched in early 2024 with backing from Andrew Ng's AI Fund, it focuses on rapid prototyping and intuitive team-based workflows.

**Documentation & Resources:**
- Official Documentation: https://docs.crewai.com/
- GitHub Repository: https://github.com/joaomdmoura/crewAI

**Mental model**: You're **hiring a team**. Define roles like researcher, writer, editor and let them collaborate.

**Best for:**
- Content pipelines (research → draft → review → publish)
- Customer support (triage → specialist → quality assurance)
- Rapid prototyping (YAML configs, minimal complex orchestration)
- Multi-role workflows with clear responsibilities

**The limitation**: Less control over edge cases. Complex workflows with 20+ conditional branches become unwieldy.

**Production adopters**: Content agencies; startups prioritizing speed-to-market; teams without extensive ML engineering resources.

### AutoGen: The Conversation Orchestrator

AutoGen focuses on LLM-to-LLM collaboration, making it powerful for experimental setups and conversational workflows (10). Developed by Microsoft Research, it emphasizes multi-agent dialogue and iterative refinement.

**Documentation & Resources:**
- Official Documentation: https://microsoft.github.io/autogen/
- GitHub Repository: https://github.com/microsoft/autogen

**Mental model**: Agents are **having a meeting**. They talk, debate, reach consensus through natural conversation.

**Best for:**
- Brainstorming systems (generate ideas → critique → refine)
- Code review workflows (developer agent ↔ QA agent dialogue)
- Exploratory research (agents collaboratively dig into topics)
- Enterprise R&D with complex collaboration requirements

**The catch**: Hard to predict behavior. Conversations can diverge without careful prompting and guard rails.

**Production adopters**: Microsoft ecosystem users; research teams; enterprises with Azure infrastructure.

### Other Notable Frameworks

**LlamaIndex Agents**: RAG-first agent capabilities over enterprise data (11)
**OpenAI Agents**: Managed runtime with first-party tools and memory
**AWS Strands Agents**: Model-first approach with MCP integration (12)
**DSPy**: Programmatic prompt optimization for reliable reasoning (8)

---

## The Decision Matrix: Choosing Your Pattern

### Pattern Selection Framework

**Choose ReAct when:**

✅ Task is exploratory/unpredictable

✅ You need real-time adaptation  

✅ Token cost is not your primary constraint

❌ Examples: Debugging, customer support, research, interactive troubleshooting

---

**Choose ReWOO when:**

✅ Task has predictable structure

✅ Token efficiency is critical (scale/cost)

✅ Your tools are reliable

❌ Examples: Multi-hop QA, data pipelines, report generation, batch processing

---

**Choose Reflexion when:**

✅ You can afford multiple attempts

✅ Task has clear success/failure signals

✅ You need learning over episodes

❌ Examples: Code generation with tests, game playing, iterative optimization

---

**Choose ToT/LATS when:**

✅ Correctness >> cost

✅ Task has explorable solution space

✅ You can verify/score outputs objectively

❌ Examples: Math reasoning, strategic planning, medical diagnosis, complex proofs

---

**Choose CodeAct when:**

✅ Tasks require computation (not just info retrieval)

✅ You have secure execution environment

✅ Output is verifiable (tests, assertions)

❌ Examples: Data analysis, scientific computing, file manipulation, automation

---

### Framework Selection (Production)

**LangGraph when:**

- Complex workflows requiring determinism

- Compliance and auditability are critical

- You need fine-grained control over state and flow

- Team has strong engineering capabilities

---

**CrewAI when:**

- Fast iteration is priority

- Role-based tasks with clear responsibilities

- Simplicity and rapid prototyping matter

- Team is smaller or less ML-focused

---

**AutoGen when:**

- Multi-agent collaboration is core

- Conversational tasks dominate

- Operating within Microsoft/Azure ecosystem

- R&D and experimentation are primary goals

---

## Emerging Trends (Late 2024 / Early 2025)

### 1. Hybrid Architectures Become Standard

The most sophisticated production systems now **combine patterns**. Common hybrids include:

- **ReWOO + ReAct fallback**: Plan with ReWOO for 80% of queries (predictable), fall back to ReAct for the 20% requiring adaptation
- **Reflexion + ToT**: Explore multiple paths with ToT, use Reflexion for iterative improvement
- **CodeAct + Verification**: Generate code solutions, verify with traditional tool calls

### 2. DSPy for Systematic Prompt Optimization

DSPy focuses on programmatic prompt optimization (8). Instead of manually fine-tuning prompts for weeks, you define high-level inputs and outputs, and DSPy compiles and optimizes the prompt automatically, often generating necessary CoT structure or few-shot examples.

This is making prompt engineering more systematic and reproducible across different models.

### 3. Cost-Aware Architecture Selection

With inference costs dropping but volume exploding, engineering teams are implementing **dynamic pattern selection**:

- Route simple queries to efficient patterns (ReWOO, direct responses)
- Escalate complex queries to adaptive patterns (ReAct, ToT)
- Use cost budgets to terminate expensive explorations

### 4. Agent-Specific Observability

Frameworks like Maxim, LangSmith, and Arize are building **agent-specific monitoring**:

- Loop detection (infinite reasoning cycles)
- Cost alerts (budget overruns)
- Faithfulness metrics (grounding in sources)
- Trajectory analysis (what paths agents actually take)

You can't improve what you don't measure, and agent observability is becoming as critical as model performance metrics.

### 5. Model Context Protocol (MCP) Standardization

MCP is emerging as a standard for tool integration - think of it as "USB for AI tools" (13). This standardization enables:

- Agents to swap tools without rewriting integration logic
- Consistent interfaces across different agent frameworks
- Easier composition of specialized tools

---

## Practical Recommendations for Your Projects

### For ML Engineers & Researchers

1. **Start with LangGraph** for understanding stateful workflows - maps directly to distributed systems concepts and event-driven architectures

2. **Study ReWOO vs ReAct** as a token optimization case study - perfect for cost analysis and efficiency research

3. **Explore CodeAct** for ML workflows - imagine agents that write preprocessing code, run experiments, and analyze results autonomously

4. **Experiment with Reflexion** for iterative improvement - could you build an agent that improves LSTM hyperparameters based on validation loss feedback?

### For Production Teams

1. **Measure before optimizing**: Implement observability first, then optimize based on actual bottlenecks

2. **Start simple, add complexity**: Begin with ReAct for flexibility, add ReWOO optimization only where cost justifies it

3. **Build hybrid systems**: Few real-world problems fit perfectly into one pattern

4. **Plan for failure**: Agents will hit edge cases - design recovery mechanisms from the start

### For Security Engineers

1. **Sandbox code execution**: If using CodeAct, isolation is non-negotiable

2. **Monitor token budgets**: Prevent denial-of-wallet attacks from infinite reasoning loops

3. **Validate tool outputs**: Don't trust external API responses blindly - agents can be manipulated through tool injection

4. **Audit trails**: Log complete agent trajectories for security review and compliance

---

## Conclusion

The landscape of agentic reasoning frameworks has matured rapidly from 2023-2025. ReAct established the foundation for adaptive agents, ReWOO proved efficiency matters at scale, Reflexion showed agents can learn from mistakes, and Tree of Thoughts demonstrated the value of exploration. CodeAct opened up entirely new action spaces through code generation.

The frameworks implementing these patterns - LangGraph, CrewAI, AutoGen, and others - each optimize for different use cases. There is no universal "best" framework; the right choice depends on your task complexity, team capabilities, and production requirements.

As we move into 2025, the trend is clear: **hybrid architectures** that combine multiple reasoning patterns, **systematic optimization** through tools like DSPy, **cost-aware routing** between patterns, and **robust observability** for production systems.

The agents of tomorrow won't just use one reasoning pattern - they'll dynamically select the right approach for each task, learn from their mistakes, and operate within well-defined safety and cost boundaries. Understanding these fundamental patterns is the foundation for building those systems.

---

## References

1. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2210.03629

2. Xu, B., Peng, Z., Lei, B., Mukherjee, S., Liu, Y., & Xu, D. (2023). ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models. *arXiv preprint arXiv:2305.18323*. https://arxiv.org/abs/2305.18323

3. IBM. (2024). What is ReWOO? IBM Think Topics. https://www.ibm.com/think/topics/rewoo

4. Shinn, N., Cassano, F., Labash, B., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *arXiv preprint arXiv:2303.11366*. https://arxiv.org/abs/2303.11366

5. CodeTree and Tree-based exploration systems. (2024). Referenced in various agent framework comparisons and implementations.

6. Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *37th Conference on Neural Information Processing Systems (NeurIPS 2023)*. https://arxiv.org/abs/2305.10601

7. Zhou, A., et al. (2023). Language Agent Tree Search Unifies Reasoning, Acting, and Planning. *arXiv preprint arXiv:2310.04406*. https://arxiv.org/abs/2310.04406

8. Capabl.in. (2024). Agentic AI Design Patterns: ReAct, ReWOO, CodeAct, and Beyond. https://capabl.in/blog/agentic-ai-design-patterns-react-rewoo-codeact-and-beyond

9. Sun, P., et al. (2024). Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation and Execution in Complex Task Handling. *arXiv preprint arXiv:2412.15305*. https://arxiv.org/abs/2412.15305

10. Singh, V. K. (2025). Battle of AI Agent Frameworks: CrewAI vs LangGraph vs AutoGen. *Medium*. https://medium.com/@vikaskumarsingh_60821/battle-of-ai-agent-frameworks-langgraph-vs-autogen-vs-crewai-3c7bf5c18979

11. DataCamp. (2025). CrewAI vs LangGraph vs AutoGen: Choosing the Right Multi-Agent AI Framework. https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen

12. Posoldova, A. (2025). Comparing 4 Agentic Frameworks: LangGraph, CrewAI, AutoGen, and Strands Agents. *Medium*. https://medium.com/@a.posoldova/comparing-4-agentic-frameworks-langgraph-crewai-autogen-and-strands-agents-b2d482691311

13. Model Context Protocol (MCP) references from LangGraph and framework integration documentation, 2024-2025.

---

## Additional Resources

### Documentation
- LangGraph: https://langchain-ai.github.io/langgraph/
- CrewAI: https://docs.crewai.com/
- AutoGen: https://microsoft.github.io/autogen/
- LangChain Agents: https://python.langchain.com/docs/modules/agents/

### Tutorials & Guides
- Prompt Engineering Guide: https://www.promptingguide.ai/
- LangGraph Tutorials: https://langchain-ai.github.io/langgraph/tutorials/
- ReAct Implementation: https://github.com/ysymyth/ReAct
- Tree of Thoughts Implementation: https://github.com/princeton-nlp/tree-of-thought-llm

### Observability & Production Tools
- LangSmith: https://www.langchain.com/langsmith
- Maxim AI: https://www.getmaxim.ai/
- Weights & Biases for LLMs: https://wandb.ai/

---

*Article by Travis (Security Engineer specializing in AI/ML)*
*Last Updated: November 2024*