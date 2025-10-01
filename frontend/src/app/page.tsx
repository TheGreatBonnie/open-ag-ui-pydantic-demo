"use client";

import { useEffect, useState } from "react";
import { PromptPanel } from "./components/prompt-panel";
import { GenerativeCanvas } from "./components/generative-canvas";
import { ComponentTree } from "./components/component-tree";
import { CashPanel } from "./components/cash-panel";
import {
  useCoAgent,
  useCoAgentStateRender,
  useCopilotAction,
  useCopilotReadable,
} from "@copilotkit/react-core";
import { BarChartComponent } from "@/app/components/chart-components/bar-chart";
import { LineChartComponent } from "@/app/components/chart-components/line-chart";
import { AllocationTableComponent } from "@/app/components/chart-components/allocation-table";
import { useCopilotChatSuggestions } from "@copilotkit/react-ui";
import { INVESTMENT_SUGGESTION_PROMPT } from "@/utils/prompts";
import { ToolLogs } from "./components/tool-logs";

export interface PortfolioState {
  id: string;
  trigger: string;
  investmentAmount?: number;
  currentPortfolioValue?: number;
  performanceData: Array<{
    date: string;
    portfolio: number;
    spy: number;
  }>;
  allocations: Array<{
    ticker: string;
    allocation: number;
    currentValue: number;
    totalReturn: number;
  }>;
  returnsData: Array<{
    ticker: string;
    return: number;
  }>;
  bullInsights: Array<{
    title: string;
    description: string;
    emoji: string;
  }>;
  bearInsights: Array<{
    title: string;
    description: string;
    emoji: string;
  }>;
  totalReturns: number;
}

export interface SandBoxPortfolioState {
  performanceData: Array<{
    date: string;
    portfolio: number;
    spy: number;
  }>;
}
export interface InvestmentPortfolio {
  ticker: string;
  amount: number;
}

export default function OpenStocksCanvas() {
  const [currentState, setCurrentState] = useState<PortfolioState>({
    id: "",
    trigger: "",
    performanceData: [],
    allocations: [],
    returnsData: [],
    bullInsights: [],
    bearInsights: [],
    currentPortfolioValue: 0,
    totalReturns: 0,
  });
  const [tempState, setTempState] = useState<any>({});
  const [sandBoxPortfolio, setSandBoxPortfolio] = useState<
    SandBoxPortfolioState[]
  >([]);
  const [selectedStock, setSelectedStock] = useState<string | null>(null);
  const [showComponentTree, setShowComponentTree] = useState(false);
  const [totalCash, setTotalCash] = useState(1000000);
  const [investedAmount, setInvestedAmount] = useState(0);

  const { state, setState } = useCoAgent({
    name: "pydanticAgent",
    initialState: {
      available_cash: totalCash,
      investment_summary: {} as any,
      investment_portfolio: [] as InvestmentPortfolio[],
      tool_logs: [],
      tools: [],
      be_stock_data: null,
      be_arguments: {},
      render_standard_charts_and_table_args: {} as any,
    },
  });

  useCoAgentStateRender({
    name: "pydanticAgent",
    render: ({ state }) =>
      state?.tool_logs?.length > 0 ? (
        <ToolLogs logs={state?.tool_logs} />
      ) : (
        <></>
      ),
  });

  useEffect(() => {
    console.log(state, "state", Object.keys(state).length);
    console.log(tempState, "tempState");
    if (Object.keys(state).length === 0) {
      setState(tempState);
    }
  }, [state, tempState]);

  useCopilotAction({
    name: "render_standard_charts_and_table",
    description:
      "This is an action to render a standard chart and table. The chart can be a bar chart or a line chart. The table can be a table of data.",
    parameters: [
      {
        name: "investment_summary",
        type: "object",
        attributes: [
          {
            name: "holdings",
            type: "object",
            description:
              "The holdings of the portfolio. a sample holding is {AAPL: 100, MSFT: 200}",
            attributes: [],
            required: true,
          },
          {
            name: "final_prices",
            type: "object",
            required: true,
            attributes: [],
            description:
              "The final prices of the holdings. a sample final price is {AAPL: 232.23, MSFT: 258.75}",
          },
          {
            name: "cash",
            type: "number",
            required: true,
            description: "The cash in the portfolio. a sample cash is 100000",
          },
          {
            name: "returns",
            type: "object",
            required: true,
            attributes: [],
            description:
              "The returns of the holdings. a sample return is { AAPL: 14252.807922363281, MSFT: 24643.160888671875}",
          },
          {
            name: "total_value",
            type: "number",
            required: true,
            description:
              "The total value of the portfolio. a sample total value is 1038895.9688110352",
          },
          {
            name: "investment_log",
            type: "object[]",
            description:
              "The investment log of the portfolio. a sample investment log is [2021-01-01: Bought 126.00 shares of AAPL at $119.02 (cost: $14996.83), 2021-01-01: Bought 88.00 shares of MSFT at $226.65 (cost: $19945.56)]",
          },
          {
            name: "add_funds_needed",
            type: "boolean",
            required: true,
            description:
              "The add funds needed flag. a sample add funds needed flag is false",
          },
          {
            name: "add_funds_dates",
            required: true,
            type: "object[]",
          },
          {
            name: "total_invested_per_stock",
            type: "object",
            required: true,
            attributes: [],
            description:
              "The total invested per stock. a sample total invested per stock is { AAPL: 14996.832000732422, MSFT: 19945.559326171875 }",
          },
          {
            name: "percent_allocation_per_stock",
            type: "object",
            required: true,
            attributes: [],
            description:
              "The percent allocation per stock. a sample percent allocation per stock is { AAPL: 42.918734039777746, MSFT: 57.08126596022225 }",
          },
          {
            name: "percent_return_per_stock",
            type: "object",
            required: true,
            attributes: [],
            description:
              "The percent return per stock. a sample percent return per stock is { AAPL: 95.03879167058213, MSFT: 123.55211746975662 }",
          },
          {
            name: "performanceData",
            type: "object[]",
            required: true,
            attributes: [
              {
                name: "date",
                type: "string",
                required: true,
                description:
                  "The date of the performance data. a sample date is 2021-01-01",
              },
              {
                name: "portfolio",
                type: "number",
                required: true,
                description:
                  "The portfolio value at the date. a sample portfolio is 34942.3913269043",
              },
              {
                name: "spy",
                type: "number",
                required: true,
                description:
                  "The spy value at the date. a sample spy is 34942.3913269043",
              },
            ],
          },
        ],
      },
      {
        name: "insights",
        type: "object",
        attributes: [
          {
            name: "bullInsights",
            type: "object[]",
            attributes: [
              {
                name: "title",
                type: "string",
                required: true,
                description:
                  "The title of the insight. a sample title is Strong Brand Loyalty",
              },
              {
                name: "description",
                type: "string",
                required: true,
                description:
                  "The description of the insight. a sample description is Apple has a dedicated customer base, which helps maintain its market dominance and drives consistent sales.",
              },
              {
                name: "emoji",
                type: "string",
                required: true,
                description: "The emoji of the insight. a sample emoji is 📈",
              },
            ],
          },
          {
            name: "bearInsights",
            type: "object[]",
            attributes: [
              {
                name: "title",
                required: true,
                type: "string",
                description:
                  "The title of the insight. a sample title is Market Volatility",
              },
              {
                name: "description",
                type: "string",
                required: true,
                description:
                  "The description of the insight. a sample description is Both Apple and Microsoft are subject to market fluctuations, which can affect investment returns.",
              },
              {
                name: "emoji",
                type: "string",
                required: true,
                description: "The emoji of the insight. a sample emoji is 📉",
              },
            ],
          },
        ],
      },
    ],
    renderAndWaitForResponse: ({ args, respond, status }) => {
      useEffect(() => {
        console.log(args, "argsargsargsargsargsaaa");
      }, [args]);
      return (
        <>
          {args?.investment_summary?.percent_allocation_per_stock &&
            args?.investment_summary?.percent_return_per_stock &&
            args?.investment_summary?.performanceData && (
              <>
                <div className="flex flex-col gap-4">
                  <LineChartComponent
                    data={args?.investment_summary?.performanceData}
                    size="small"
                  />
                  <BarChartComponent
                    data={Object.entries(
                      args?.investment_summary?.percent_return_per_stock
                    ).map(([ticker, return1]) => ({
                      ticker,
                      return: return1 as number,
                    }))}
                    size="small"
                  />
                  <AllocationTableComponent
                    allocations={Object.entries(
                      args?.investment_summary?.percent_allocation_per_stock
                    ).map(([ticker, allocation]) => ({
                      ticker,
                      allocation: allocation as number,
                      currentValue:
                        (args?.investment_summary?.final_prices as any)[
                          ticker
                        ] * (args?.investment_summary?.holdings as any)[ticker],
                      totalReturn: (
                        args?.investment_summary
                          ?.percent_return_per_stock as any
                      )[ticker],
                    }))}
                    size="small"
                  />
                </div>

                <button
                  hidden={status == "complete"}
                  className="mt-4 rounded-full px-6 py-2 bg-green-50 text-green-700 border border-green-200 shadow-sm hover:bg-green-100 transition-colors font-semibold text-sm"
                  onClick={() => {
                    debugger;
                    if (respond) {
                      setTotalCash(args?.investment_summary?.cash);
                      setCurrentState({
                        ...currentState,
                        returnsData: Object.entries(
                          args?.investment_summary?.percent_return_per_stock
                        ).map(([ticker, return1]) => ({
                          ticker,
                          return: return1 as number,
                        })),
                        allocations: Object.entries(
                          args?.investment_summary?.percent_allocation_per_stock
                        ).map(([ticker, allocation]) => ({
                          ticker,
                          allocation: allocation as number,
                          currentValue:
                            (args?.investment_summary?.final_prices as any)[
                              ticker
                            ] *
                            (args?.investment_summary?.holdings as any)[ticker],
                          totalReturn: (
                            args?.investment_summary
                              ?.percent_return_per_stock as any
                          )[ticker],
                        })),
                        performanceData:
                          args?.investment_summary?.performanceData,
                        bullInsights: args?.insights?.bullInsights || [],
                        bearInsights: args?.insights?.bearInsights || [],
                        currentPortfolioValue:
                          args?.investment_summary?.total_value,
                        totalReturns: (
                          Object.values(
                            args?.investment_summary?.returns
                          ) as number[]
                        ).reduce((acc, val) => acc + val, 0),
                      });
                      setInvestedAmount(
                        (
                          Object.values(
                            args?.investment_summary?.total_invested_per_stock
                          ) as number[]
                        ).reduce((acc, val) => acc + val, 0)
                      );
                      setTempState({
                        ...state,
                        available_cash: args?.investment_summary?.cash,
                      });
                      // setState({
                      //   ...state,
                      //   available_cash: args?.investment_summary?.cash,
                      // })
                      respond(
                        "Data rendered successfully. Provide summary of the investments by not making any tool calls"
                      );
                    }
                  }}>
                  Accept
                </button>
                <button
                  hidden={status == "complete"}
                  className="rounded-full px-6 py-2 bg-red-50 text-red-700 border border-red-200 shadow-sm hover:bg-red-100 transition-colors font-semibold text-sm ml-2"
                  onClick={() => {
                    debugger;
                    if (respond) {
                      respond(
                        "Data rendering rejected. Just give a summary of the rejected investments by not making any tool calls"
                      );
                    }
                  }}>
                  Reject
                </button>
              </>
            )}
        </>
      );
    },
  });

  // useCopilotAction({
  //   name: "render_custom_charts",
  //   renderAndWaitForResponse: ({ args, respond, status }) => {
  //     return (
  //       <>
  //         <LineChartComponent data={args?.investment_summary?.performanceData} size="small" />
  //         <button hidden={status == "complete"}
  //           className="mt-4 rounded-full px-6 py-2 bg-green-50 text-green-700 border border-green-200 shadow-sm hover:bg-green-100 transition-colors font-semibold text-sm"
  //           onClick={() => {
  //             debugger
  //             if (respond) {
  //               setSandBoxPortfolio([...sandBoxPortfolio, {
  //                 performanceData: args?.investment_summary?.performanceData.map((item: any) => ({
  //                   date: item.date,
  //                   portfolio: item.portfolio,
  //                   spy: 0
  //                 })) || []
  //               }])
  //               respond("Data rendered successfully. Provide summary of the investments")
  //             }
  //           }}
  //         >
  //           Accept
  //         </button>
  //         <button hidden={status == "complete"}
  //           className="rounded-full px-6 py-2 bg-red-50 text-red-700 border border-red-200 shadow-sm hover:bg-red-100 transition-colors font-semibold text-sm ml-2"
  //           onClick={() => {
  //             debugger
  //             if (respond) {
  //               respond("Data rendering rejected. Just give a summary of the rejected investments")
  //             }
  //           }}
  //         >
  //           Reject
  //         </button>
  //       </>
  //     )
  //   }
  // })

  useCopilotReadable({
    description: "This is the current state of the portfolio",
    value: JSON.stringify(state.investment_portfolio),
  });

  useCopilotChatSuggestions(
    {
      available: selectedStock ? "disabled" : "enabled",
      instructions: INVESTMENT_SUGGESTION_PROMPT,
    },
    [selectedStock]
  );

  // const toggleComponentTree = () => {
  //   setShowComponentTree(!showComponentTree)
  // }

  // const availableCash = totalCash - investedAmount
  // const currentPortfolioValue = currentState.currentPortfolioValue || investedAmount

  useEffect(() => {
    getBenchmarkData();
  }, []);

  function getBenchmarkData() {
    let result: PortfolioState = {
      id: "aapl-nvda",
      trigger: "apple nvidia",
      performanceData: [
        { date: "Jan 2023", portfolio: 10000, spy: 10000 },
        { date: "Mar 2023", portfolio: 10200, spy: 10200 },
        { date: "Jun 2023", portfolio: 11000, spy: 11000 },
        { date: "Sep 2023", portfolio: 10800, spy: 10800 },
        { date: "Dec 2023", portfolio: 11500, spy: 11500 },
        { date: "Mar 2024", portfolio: 12200, spy: 12200 },
        { date: "Jun 2024", portfolio: 12800, spy: 12800 },
        { date: "Sep 2024", portfolio: 13100, spy: 13100 },
        { date: "Dec 2024", portfolio: 13600, spy: 13600 },
      ],
      allocations: [],
      returnsData: [],
      bullInsights: [],
      bearInsights: [],
      totalReturns: 0,
      currentPortfolioValue: totalCash,
    };
    setCurrentState(result);
  }

  return (
    <div className="h-screen bg-[#FAFCFA] flex overflow-hidden">
      {/* Left Panel - Prompt Input */}
      <div className="w-85 border-r border-[#D8D8E5] bg-white flex-shrink-0">
        <PromptPanel availableCash={totalCash} />
      </div>

      {/* Center Panel - Generative Canvas */}
      <div className="flex-1 relative min-w-0">
        {/* Top Bar with Cash Info */}
        <div className="absolute top-0 left-0 right-0 bg-white border-b border-[#D8D8E5] p-4 z-10">
          <CashPanel
            totalCash={totalCash}
            investedAmount={investedAmount}
            currentPortfolioValue={
              totalCash + investedAmount + currentState.totalReturns || 0
            }
            onTotalCashChange={setTotalCash}
            onStateCashChange={setState}
          />
        </div>

        {/* <div className="absolute top-4 right-4 z-20">
          <button
            onClick={toggleComponentTree}
            className="px-3 py-1 text-xs font-semibold text-[#575758] bg-white border border-[#D8D8E5] rounded-md hover:bg-[#F0F0F4] transition-colors"
          >
            {showComponentTree ? "Hide Tree" : "Show Tree"}
          </button>
        </div> */}

        <div className="pt-20 h-full">
          <GenerativeCanvas
            setSelectedStock={setSelectedStock}
            portfolioState={currentState}
            sandBoxPortfolio={sandBoxPortfolio}
            setSandBoxPortfolio={setSandBoxPortfolio}
          />
        </div>
      </div>

      {/* Right Panel - Component Tree (Optional) */}
      {showComponentTree && (
        <div className="w-64 border-l border-[#D8D8E5] bg-white flex-shrink-0">
          <ComponentTree portfolioState={currentState} />
        </div>
      )}
    </div>
  );
}
