<p align="center">
  <img src="Paper Architecture notes.png" alt="research paper architecture explained" width="100%">
</p>

# CRAG-LangGraph-Research-Implementation
Research implementation of Corrective Retrieval-Augmented Generation (CRAG) using LangGraph, featuring retrieval evaluation, conditional correction, knowledge refinement, web augmentation, and grounded generation.

# Official Paper link

- https://arxiv.org/pdf/2401.15884

# Graph flow:

                    ┌──────────────┐
                    │  Retriever   │
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │  Evaluator   │
                    └──────┬───────┘
                           │
             ┌─────────────┼─────────────┐
             ↓             ↓             ↓
         correct       ambiguous      incorrect
             │             │             │
             ↓             └──────┬──────┘
       refine_docs                ↓
             │                web_search
             ↓                    ↓
    generate_from_docs       tavily_tool
                                 ↓
                         refine_web_results
                                 │
                         ┌───────┴────────┐
                         │                │
                    ambiguous        incorrect
                         │                │
                         ↓                ↓
              generate_from_both   generate_from_web

