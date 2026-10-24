# Multi-Agent Orchestration System

## Project Overview
This project implements a production-grade, enterprise-level multi-agent AI system. Built using Amazon Bedrock AgentCore and the Strands Agents SDK, the system orchestrates five specialized AI agents to handle complex requests, demonstrating advanced AI routing, parallel data retrieval, and enterprise safety guardrails.

## System Architecture
The system uses a Hub-and-Spoke (Orchestrator-Worker) architecture:
1. Orchestrator Agent: Analyzes prompts, initializes session state, and delegates tasks.
2. Inventory Agent: Queries Amazon DynamoDB for order statuses and customer tiers.
3. Refund Agent: Applies business logic (tier-based return windows) using optimistic locking.
4. Policy Agent: Uses parallel RAG to query three Amazon Bedrock Knowledge Bases simultaneously.
5. Communication Agent: Synthesizes all facts into a professional, empathetic response.

## Key Features
- Multi-Agent Orchestration: Specialized agents working together via a central router.
- Optimistic Locking: DynamoDB version control ensures data integrity.
- Parallel RAG: Concurrent querying of multiple Knowledge Bases for sub-second retrieval.
- Enterprise Guardrails: Safety filters blocking PII, competitor mentions, and restricted topics.
- Observability: Distributed tracing via AWS X-Ray and centralized logging via Amazon CloudWatch.

## Project Structure
- README.md: This documentation file.
- submission_screenshots/: Proof of successful deployment and testing.
- project/starter/src/: Core multi-agent implementation.
- project/starter/tests/: Automated grading suite.
- project/starter/config.py: AWS resource configuration.
- project/starter/.env: Environment variables.
- project/starter/infrastructure/: CloudFormation templates and seed scripts.

Built as a demonstration of enterprise AI architecture and multi-agent orchestration.
