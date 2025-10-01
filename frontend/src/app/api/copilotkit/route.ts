import {
  CopilotRuntime,
  copilotRuntimeNextJSAppRouterEndpoint,
  OpenAIAdapter
} from '@copilotkit/runtime';
import { NextRequest } from 'next/server';
import { HttpAgent } from "@ag-ui/client";


const pydanticAgent = new HttpAgent({
  url: process.env.NEXT_PUBLIC_PYDANTIC_URL || "http://0.0.0.0:8000/pydantic-agent",
});
const serviceAdapter = new OpenAIAdapter()
const runtime = new CopilotRuntime({
  agents: {
    // @ts-ignore
    pydanticAgent : pydanticAgent 
  },
});
// const runtime = new CopilotRuntime()
export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: '/api/copilotkit',
  });

  return handleRequest(req);
};