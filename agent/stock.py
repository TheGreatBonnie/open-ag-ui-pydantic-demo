# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

# Pydantic imports for data validation and modeling
from pydantic import BaseModel, Field
from typing import Any, Literal

# AG-UI core event types for managing application state and UI updates
from ag_ui.core import (
    StateSnapshotEvent,      # For capturing complete state snapshots
    EventType,               # Event type enumeration
    StateDeltaEvent,         # For incremental state changes
    TextMessageStartEvent,   # For chat message events
    TextMessageEndEvent,
    TextMessageContentEvent,
)

# Pydantic AI imports for agent creation and context management
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps

# Utility imports
from dotenv import load_dotenv  # For loading environment variables
import uuid                     # For generating unique identifiers
from datetime import datetime   # For date/time operations
from textwrap import dedent     # For formatted string handling

# Financial data and analysis libraries
import yfinance as yf          # For fetching stock market data
import numpy as np             # For numerical computations
import pandas as pd            # For data manipulation and analysis

# Load environment variables from .env file (e.g., API keys)
load_dotenv()


# ============================================================================
# DATA MODELS
# ============================================================================

class AgentState(BaseModel):
    """
    Main application state that holds all the data for the stock portfolio agent.
    This state is managed throughout the agent's lifecycle and updated by various tools.
    """
    tools: list = []                                    # List of available tools for the agent
    be_stock_data: Any = None                          # Backend stock price data (dictionary format)
    be_arguments: dict = {}                            # Arguments passed to backend operations
    available_cash: float = 0.0                       # Amount of cash available for investment
    investment_summary: dict = {}                      # Summary of investment performance and holdings
    investment_portfolio: list = []                    # List of stocks in the portfolio with amounts
    tool_logs: list = []                              # Log of tool executions for debugging
    render_standard_charts_and_table_args: dict = {}  # Arguments for rendering charts and tables


class JSONPatchOp(BaseModel):
    """
    A class representing a JSON Patch operation (RFC 6902).
    Used for making incremental updates to the application state.
    """
    op: Literal["add", "remove", "replace", "move", "copy", "test"] = Field(
        description="The operation to perform: add, remove, replace, move, copy, or test",
    )
    path: str = Field(description="JSON Pointer (RFC 6901) to the target location")
    value: Any = Field(
        default=None,
        description="The value to apply (for add, replace operations)",
    )
    from_: str | None = Field(
        default=None,
        alias="from",
        description="Source path (for move, copy operations)",
    )


class bull_insights(BaseModel):
    """
    Data model for positive/bullish insights about a stock.
    These insights highlight potential opportunities and strengths.
    """
    title: str = Field(description="Short title for the positive insight.")
    description: str = Field(
        description="Detailed description of the positive insight."
    )
    emoji: str = Field(description="Emoji representing the positive insight.")


class bear_insights(BaseModel):
    """
    Data model for negative/bearish insights about a stock.
    These insights highlight potential risks and weaknesses.
    """
    title: str = Field(description="Short title for the negative insight.")
    description: str = Field(
        description="Detailed description of the negative insight."
    )
    emoji: str = Field(description="Emoji representing the negative insight.")


class insights(BaseModel):
    """
    Container model that groups both bullish and bearish insights together.
    Used for organizing and passing insights data between functions.
    """
    bullInsights: list[bull_insights]  # List of positive insights
    bearInsights: list[bear_insights]  # List of negative insights


# ============================================================================
# AGENT INITIALIZATION
# ============================================================================

# Create the main AI agent using OpenAI's GPT-4o-mini model
# The agent will handle stock portfolio analysis and investment operations
agent = Agent(
    "openai:gpt-4o-mini",  # Specify the AI model to use
    deps_type=StateDeps[AgentState],  # Specify the state dependency type
)


@agent.instructions
async def instructions(ctx: RunContext[StateDeps[AgentState]]) -> str:
    """
    Dynamic instructions for the agent that can access the current state context.
    These instructions guide the agent's behavior and tool usage patterns.
    
    Args:
        ctx: The current run context containing the agent state
        
    Returns:
        str: Formatted instructions for the agent
    """
    return dedent(f"""You are a stock portfolio analysis agent. 
                  Use the tools provided effectively to answer the user query.
                  When user asks something related to the stock investment, make 
                  sure to call the frontend tool render_standard_charts_and_table 
                  with the tool argument render_standard_charts_and_table_args as 
                  the tool argument to the frontend after running the generate_insights tool""")


# ============================================================================
# AGENT TOOLS - Stock Data Management
# ============================================================================


@agent.tool
async def gather_stock_data(
    ctx: RunContext[StateDeps[AgentState]],
    stock_tickers_list: list[str],
    investment_date: str,
    interval_of_investment: str,
    amount_of_dollars_to_be_invested: list[float],
    operation: Literal["add", "replace", "delete"],
    to_be_replaced : list[str]
) -> list[StateSnapshotEvent, StateDeltaEvent]:
    """
    Gathers historical stock data and manages the investment portfolio based on the specified operation.
    This is the primary tool for setting up and modifying stock portfolios.
    
    This tool is used for the chat purposes. If the user query is not related to the stock portfolio, 
    you should use this tool to answer the question. The answers should be generic and should be relevant to the user query.

    Args:
        ctx: The current run context containing the agent state
        stock_tickers_list: List of stock ticker symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])
        investment_date: Starting date for the investment in YYYY-MM-DD format
        interval_of_investment: Frequency of investment (currently supports "single_shot")
        amount_of_dollars_to_be_invested: List of dollar amounts to invest in each corresponding stock
        operation: Type of portfolio operation - "add" (append to existing), "replace" (override existing), or "delete" (remove specified)
        to_be_replaced: List of tickers to be replaced (used with "replace" operation)

    Returns:
        list: Contains StateSnapshotEvent and StateDeltaEvent for updating the UI state
    """
    # Debug print to monitor function parameters
    print(
        stock_tickers_list,
        investment_date,
        interval_of_investment,
        amount_of_dollars_to_be_invested,
        operation,
    )
    
    # Initialize list to track state changes for the UI
    changes = []
    
    # Add initial tool log entry to show the tool has started
    tool_log_start_id = str(uuid.uuid4())
    changes.append(
        JSONPatchOp(
            op="add",
            path="/tool_logs/-",
            value={
                "message": "Starting stock data gathering...",
                "status": "in_progress",
                "id": tool_log_start_id,
            },
        )
    )

    # STEP 1: Handle portfolio operations based on existing investments
    if len(ctx.deps.state.investment_portfolio) > 0:
        
        # ADD OPERATION: Append new stocks to existing portfolio
        if(operation == "add"):
            for i in ctx.deps.state.investment_portfolio:
                stock_tickers_list.append(i["ticker"])                    # Add existing ticker
                amount_of_dollars_to_be_invested.append(i["amount"])      # Add existing amount
        
        # DELETE OPERATION: Remove specified stocks from portfolio
        if(operation == "delete"):
            for i in ctx.deps.state.investment_portfolio:
                if i["ticker"] in stock_tickers_list:
                    stock_tickers_list.remove(i["ticker"])               # Remove if in delete list
                    # amount_of_dollars_to_be_invested.remove(i["amount"])  # (commented out)
                else:
                    stock_tickers_list.append(i["ticker"])               # Keep if not in delete list
                    amount_of_dollars_to_be_invested.append(i["amount"]) # Keep existing amount
        
        # REPLACE OPERATION: Replace existing portfolio with new specifications
        if(operation == "replace"):
            items = []
            amounts = []
            # First, collect all existing portfolio items
            for i in ctx.deps.state.investment_portfolio:
                # Note: Replacement logic is commented out for now
                # if i["ticker"] in to_be_replaced:
                #     i['ticker'] = to_be_replaced[to_be_replaced.index(i["ticker"])]
                items.append(i["ticker"])
                amounts.append(i["amount"])
            
            # Then, add any new items from to_be_replaced list that aren't already present
            for i in to_be_replaced:
                if i not in items:
                    items.append(i)
                    amounts.append(0)  # Default amount for new items
            
            # Update the working lists with the replacement data
            # Alternative approach (commented out):
            # for i in range(len(items)):
            #     stock_tickers_list.append(items[i])
            #     amount_of_dollars_to_be_invested.append(amounts[i])
            stock_tickers_list = items
            amount_of_dollars_to_be_invested = amounts

    # Debug print to verify the final ticker list and amounts
    print(stock_tickers_list, amount_of_dollars_to_be_invested,"stock_tickers_list, amount_of_dollars_to_be_invested")
    
    # STEP 2: Update the investment portfolio in the application state
    changes.append(
        JSONPatchOp(
            op="replace",
            path="/investment_portfolio",
            value=[
                {
                    "ticker": ticker,
                    "amount": amount_of_dollars_to_be_invested[index],
                }
                for index, ticker in enumerate(stock_tickers_list)
            ],
        )
    )
    
    # Update the local state object as well (for immediate use in this function)
    ctx.deps.state.investment_portfolio = [
        {
            "ticker": ticker,
            "amount": amount_of_dollars_to_be_invested[index],
        }
        for index, ticker in enumerate(stock_tickers_list)
    ]
    
    # STEP 3: Prepare data for Yahoo Finance API call
    tickers = stock_tickers_list
    investment_date = investment_date
    current_year = datetime.now().year
    
    # Validate and adjust investment date (Yahoo Finance has data limitations)
    # If investment date is more than 4 years ago, adjust it to prevent API issues
    if current_year - int(investment_date[:4]) > 4:
        print("investment date is more than 4 years ago")
        investment_date = f"{current_year - 4}-01-01"  # Set to 4 years ago maximum
    
    # Determine the appropriate history period for data retrieval
    if current_year - int(investment_date[:4]) == 0:
        history_period = "1y"  # If current year, get 1 year of data
    else:
        history_period = f"{current_year - int(investment_date[:4])}y"  # Otherwise, get N years

    # STEP 4: Fetch stock data from Yahoo Finance
    # Download historical stock prices for all tickers
    data = yf.download(
        tickers,                                      # List of stock symbols
        period=history_period,                        # Time period to fetch
        interval="3mo",                               # Data frequency (quarterly)
        start=investment_date,                        # Start date
        end=datetime.today().strftime("%Y-%m-%d"),    # End date (today)
    )
    
    # STEP 5: Store stock data in application state
    # Convert the closing prices to dictionary format for easier handling
    changes.append(
        JSONPatchOp(
            op="replace",
            path="/be_stock_data",
            value=data["Close"].to_dict(),  # Extract closing prices and convert to dict
        )
    )
    ctx.deps.state.be_stock_data = data["Close"].to_dict()  # Update local state
    
    # STEP 6: Store the arguments used for this data gathering operation
    changes.append(
        JSONPatchOp(
            op="replace",
            path="/be_arguments",
            value={
                "ticker_symbols": stock_tickers_list,
                "investment_date": investment_date,
                "amount_of_dollars_to_be_invested": amount_of_dollars_to_be_invested,
                "interval_of_investment": interval_of_investment,
            },
        )
    )
    
    # Generate unique ID for completion tool logging
    tool_log_id = str(uuid.uuid4())
    # Add completion tool log entry for data gathering
    changes.append(
        JSONPatchOp(
            op="add",
            path="/tool_logs/-",
            value={
                "message": "Stock data gathering completed successfully",
                "status": "completed",
                "id": tool_log_id,
            },
        )
    )
    
    # Update local state with the backend arguments
    ctx.deps.state.be_arguments = {
        "ticker_symbols": stock_tickers_list,
        "investment_date": investment_date,
        "amount_of_dollars_to_be_invested": amount_of_dollars_to_be_invested,
        "interval_of_investment": interval_of_investment,
    }
    
    # Debug print to verify the final state
    print((ctx.deps.state).model_dump(),"ctx.deps.statectx.deps.state")
    
    # STEP 7: Return state events for UI updates
    return [
        # Complete state snapshot for full UI refresh
        StateSnapshotEvent(
            type=EventType.STATE_SNAPSHOT,
            snapshot=(ctx.deps.state).model_dump(),
        ),
        # Incremental changes for efficient UI updates
        StateDeltaEvent(type=EventType.STATE_DELTA, delta=changes),
    ]


# ============================================================================
# AGENT TOOLS - Cash Allocation and Portfolio Analysis
# ============================================================================


@agent.tool
async def allocate_cash(
    ctx: RunContext[StateDeps[AgentState]],
) -> list[StateDeltaEvent]:
    """
    Allocates cash to stocks based on the portfolio data and calculates investment performance.
    This function simulates the investment process and compares portfolio performance with SPY.
    
    This tool should be called after gather_stock_data so as to allocate cash to respective stocks 
    extracted from previous stock data gathering operation.

    Args:
        ctx: The current run context containing the agent state with stock data

    Returns:
        list[StateDeltaEvent]: List containing state change events for the UI
    """
    
    # STEP 1: Retrieve and prepare stock data
    # Get the stock data dictionary from the previous gather_stock_data operation
    stock_data_dict = (
        ctx.deps.state.be_stock_data
    )  # Dictionary format: {ticker: {date: price}}
    
    # Generate unique ID for starting log entry
    tool_log_start_id = str(uuid.uuid4())
    
    # Convert the dictionary back to a pandas DataFrame for easier processing
    stock_data = pd.DataFrame(stock_data_dict)
    
    # Extract the arguments from the previous operation
    args = ctx.deps.state.be_arguments
    tickers = args["ticker_symbols"]                    # List of stock tickers
    investment_date = args["investment_date"]           # Investment start date
    amounts = args["amount_of_dollars_to_be_invested"]  # List of amounts per ticker
    interval = "single_shot"                           # Investment strategy type

    # STEP 2: Initialize investment parameters
    # Determine total available cash for investment
    if ctx.deps.state.available_cash is not None:
        total_cash = ctx.deps.state.available_cash      # Use existing cash if available
    else:
        total_cash = sum(amounts)                       # Otherwise, sum of all amounts
    
    # Initialize tracking variables
    holdings = {ticker: 0.0 for ticker in tickers}     # Track shares owned per ticker
    investment_log = []                                 # Log of all investment transactions
    add_funds_needed = False                           # Flag for insufficient funds
    add_funds_dates = []                               # Dates when more funds were needed

    # STEP 3: Ensure data is properly sorted by date
    stock_data = stock_data.sort_index()

    # STEP 4: Execute investment strategy
    if interval == "single_shot":
        """
        SINGLE SHOT INVESTMENT STRATEGY:
        Buy all shares at the first available date using pre-allocated money for each ticker.
        This simulates investing all money at once rather than dollar-cost averaging.
        """
        # Get the first trading date from the stock data
        first_date = stock_data.index[0]
        row = stock_data.loc[first_date]  # Stock prices on the first date
        
        # Iterate through each ticker and attempt to purchase shares
        for idx, ticker in enumerate(tickers):
            price = row[ticker]                          # Stock price on first date
            
            # Handle missing price data
            if np.isnan(price):
                investment_log.append(
                    f"{first_date.date()}: No price data for {ticker}, could not invest."
                )
                add_funds_needed = True
                add_funds_dates.append(
                    (str(first_date.date()), ticker, price, amounts[idx])
                )
                continue
            
            # Calculate investment for this specific ticker
            allocated = amounts[idx]                     # Amount allocated to this ticker
            
            # Check if we have enough cash and the allocation is sufficient
            if total_cash >= allocated and allocated >= price:
                # Calculate how many whole shares we can buy
                shares_to_buy = allocated // price       # Floor division for whole shares
                
                if shares_to_buy > 0:
                    # Execute the purchase
                    cost = shares_to_buy * price         # Total cost of purchase
                    holdings[ticker] += shares_to_buy    # Add shares to holdings
                    total_cash -= cost                   # Deduct cost from available cash
                    
                    # Log the successful transaction
                    investment_log.append(
                        f"{first_date.date()}: Bought {shares_to_buy:.2f} shares of {ticker} at ${price:.2f} (cost: ${cost:.2f})"
                    )
                else:
                    # Not enough allocated cash for even one share
                    investment_log.append(
                        f"{first_date.date()}: Not enough allocated cash to buy {ticker} at ${price:.2f}. Allocated: ${allocated:.2f}"
                    )
                    add_funds_needed = True
                    add_funds_dates.append(
                        (str(first_date.date()), ticker, price, allocated)
                    )
            else:
                # Insufficient total cash or allocation
                investment_log.append(
                    f"{first_date.date()}: Not enough total cash to buy {ticker} at ${price:.2f}. Allocated: ${allocated:.2f}, Available: ${total_cash:.2f}"
                )
                add_funds_needed = True
                add_funds_dates.append(
                    (str(first_date.date()), ticker, price, total_cash)
                )
        # No further purchases on subsequent dates in single_shot mode
    else:
        """
        ALTERNATIVE INVESTMENT STRATEGY (Dollar Cost Averaging):
        This section handles other investment intervals like DCA, but is not currently used.
        """
        # DCA or other interval logic (previous logic)
        for date, row in stock_data.iterrows():
            for i, ticker in enumerate(tickers):
                price = row[ticker]
                if np.isnan(price):
                    continue  # skip if price is NaN
                # Invest as much as possible for this ticker at this date
                if total_cash >= price:
                    shares_to_buy = total_cash // price
                    if shares_to_buy > 0:
                        cost = shares_to_buy * price
                        holdings[ticker] += shares_to_buy
                        total_cash -= cost
                        investment_log.append(
                            f"{date.date()}: Bought {shares_to_buy:.2f} shares of {ticker} at ${price:.2f} (cost: ${cost:.2f})"
                        )
                else:
                    add_funds_needed = True
                    add_funds_dates.append(
                        (str(date.date()), ticker, price, total_cash)
                    )
                    investment_log.append(
                        f"{date.date()}: Not enough cash to buy {ticker} at ${price:.2f}. Available: ${total_cash:.2f}. Please add more funds."
                    )

    # STEP 5: Calculate portfolio performance metrics
    # Get the most recent stock prices (last row in the data)
    final_prices = stock_data.iloc[-1]
    total_value = 0.0                                    # Total portfolio value
    returns = {}                                         # Absolute returns per stock
    total_invested_per_stock = {}                        # Amount invested per stock
    percent_allocation_per_stock = {}                    # Percentage allocation per stock
    percent_return_per_stock = {}                        # Percentage return per stock
    total_invested = 0.0                                 # Total amount invested across all stocks
    
    # Calculate investment amounts for each ticker
    for idx, ticker in enumerate(tickers):
        # Calculate how much was actually invested in this stock
        if interval == "single_shot":
            # Only one purchase at first date - calculate based on actual holdings
            first_date = stock_data.index[0]
            price = stock_data.loc[first_date][ticker]
            shares_bought = holdings[ticker]
            invested = shares_bought * price             # Actual amount invested
        else:
            # Alternative approach: Sum all purchases from the investment log
            # Parse the investment log to calculate total invested per stock
            invested = 0.0
            for log in investment_log:
                if f"shares of {ticker}" in log and "Bought" in log:
                    # Extract cost from log string (format: "cost: $X.XX")
                    try:
                        cost_str = log.split("(cost: $")[-1].split(")")[0]
                        invested += float(cost_str)
                    except Exception:
                        pass  # Skip if parsing fails
        
        # Store the invested amount for this ticker
        total_invested_per_stock[ticker] = invested
        total_invested += invested                       # Add to total invested amount
    
    # STEP 6: Calculate percentage allocations and returns
    # Now that we have total_invested, calculate percentages and returns
    for ticker in tickers:
        invested = total_invested_per_stock[ticker]      # Amount invested in this ticker
        holding_value = holdings[ticker] * final_prices[ticker]  # Current value of holdings
        
        # Calculate absolute return (profit/loss in dollars)
        returns[ticker] = holding_value - invested
        total_value += holding_value                     # Add to total portfolio value
        
        # Calculate percentage allocation (what % of total investment went to this stock)
        percent_allocation_per_stock[ticker] = (
            (invested / total_invested * 100) if total_invested > 0 else 0.0
        )
        
        # Calculate percentage return (what % gain/loss on this investment)
        percent_return_per_stock[ticker] = (
            ((holding_value - invested) / invested * 100) if invested > 0 else 0.0
        )
    
    total_value += total_cash  # Add remaining uninvested cash to total portfolio value

    # STEP 7: Store investment results in application state
    # Package all the calculated results into the investment summary
    ctx.deps.state.investment_summary = {
        "holdings": holdings,                              # Shares owned per ticker
        "final_prices": final_prices.to_dict(),           # Current stock prices
        "cash": total_cash,                                # Remaining uninvested cash
        "returns": returns,                                # Absolute returns per ticker
        "total_value": total_value,                        # Total portfolio value
        "investment_log": investment_log,                  # Transaction history
        "add_funds_needed": add_funds_needed,              # Whether more funds are needed
        "add_funds_dates": add_funds_dates,                # Specific dates/stocks needing funds
        "total_invested_per_stock": total_invested_per_stock,      # Investment per stock
        "percent_allocation_per_stock": percent_allocation_per_stock,  # Allocation percentages
        "percent_return_per_stock": percent_return_per_stock,      # Return percentages
    }
    ctx.deps.state.available_cash = float(total_cash)  # Update available cash in state

    # ========================================================================
    # STEP 8: BENCHMARK COMPARISON - Portfolio vs SPY Performance
    # ========================================================================
    """
    This section compares the portfolio's performance against the S&P 500 (SPY ETF).
    It simulates investing the same total amount in SPY and tracks performance over time.
    """
    
    # Get SPY (S&P 500 ETF) prices for the same time period
    spy_ticker = "SPY"
    spy_prices = None
    try:
        # Download SPY data for the same period as our stock data
        spy_prices = yf.download(
            spy_ticker,
            # Calculate period length based on stock data length
            period=f"{len(stock_data)//4}y" if len(stock_data) > 4 else "1y",
            interval="3mo",                                # Same interval as stock data
            start=stock_data.index[0],                     # Same start date
            end=stock_data.index[-1],                      # Same end date
        )["Close"]
        
        # Align SPY prices to our stock data dates using forward fill
        # This ensures we have SPY prices for the exact same dates as our portfolio
        spy_prices = spy_prices.reindex(stock_data.index, method="ffill")
    except Exception as e:
        print("Error fetching SPY data:", e)
        # Create dummy SPY prices if fetch fails
        spy_prices = pd.Series([None] * len(stock_data), index=stock_data.index)

    # STEP 9: Simulate investing the same amount in SPY
    # Initialize SPY investment tracking variables
    spy_shares = 0.0                                     # Total SPY shares owned
    spy_cash = total_invested                            # Cash to invest in SPY (same as portfolio)
    spy_invested = 0.0                                   # Actual amount invested in SPY
    spy_investment_log = []                              # Log of SPY transactions
    
    if interval == "single_shot":
        """
        SPY Single Shot Investment: Invest all money in SPY at the first date
        """
        first_date = stock_data.index[0]
        spy_price = spy_prices.loc[first_date]
        
        # Handle potential Series vs scalar price data
        if isinstance(spy_price, pd.Series):
            spy_price = spy_price.iloc[0]                # Extract scalar value from Series
        
        # Execute SPY purchase if price data is available
        if not pd.isna(spy_price):
            spy_shares = spy_cash // spy_price           # Calculate shares to buy
            spy_invested = spy_shares * spy_price        # Calculate actual investment
            spy_cash -= spy_invested                     # Deduct investment from available cash
            
            # Log the SPY transaction
            spy_investment_log.append(
                f"{first_date.date()}: Bought {spy_shares:.2f} shares of SPY at ${spy_price:.2f} (cost: ${spy_invested:.2f})"
            )
    else:
        """
        SPY Dollar Cost Averaging: Invest equal portions at each date
        """
        dca_amount = total_invested / len(stock_data)    # Amount to invest per period
        
        # Invest the DCA amount at each date in our data
        for date in stock_data.index:
            spy_price = spy_prices.loc[date]
            
            # Handle potential Series vs scalar price data
            if isinstance(spy_price, pd.Series):
                spy_price = spy_price.iloc[0]
            
            # Execute SPY purchase for this period
            if not pd.isna(spy_price):
                shares = dca_amount // spy_price         # Shares to buy this period
                cost = shares * spy_price                # Cost for this period
                spy_shares += shares                     # Add to total shares
                spy_cash -= cost                         # Deduct cost from cash
                spy_invested += cost                     # Add to total invested
                
                # Log this SPY transaction
                spy_investment_log.append(
                    f"{date.date()}: Bought {shares:.2f} shares of SPY at ${spy_price:.2f} (cost: ${cost:.2f})"
                )

    # STEP 10: Build performance comparison data over time
    """
    Create a time series comparing portfolio value vs SPY value at each date.
    This data will be used for charting performance over time.
    """
    performanceData = []
    running_holdings = holdings.copy()                   # Copy of final holdings (static for historical calc)
    running_cash = total_cash                            # Remaining cash after all investments
    
    # Calculate portfolio and SPY values at each date in our dataset
    for date in stock_data.index:
        # PORTFOLIO VALUE CALCULATION
        # Sum of (shares owned * stock price at this date) for all stocks
        port_value = (
            sum(
                running_holdings[t] * stock_data.loc[date][t]  # shares * price
                for t in tickers
                if not pd.isna(stock_data.loc[date][t])        # Only include valid prices
            )
            # Note: Cash is commented out here since we're tracking investment performance
            # + running_cash
        )
        
        # SPY VALUE CALCULATION
        # SPY shares owned * SPY price at this date + remaining cash
        spy_price = spy_prices.loc[date]
        if isinstance(spy_price, pd.Series):
            spy_price = spy_price.iloc[0]                # Extract scalar from Series
        
        # Calculate SPY portfolio value (shares * price + cash)
        spy_val = spy_shares * spy_price + spy_cash if not pd.isna(spy_price) else None
        
        # Add this date's performance data to our time series
        performanceData.append(
            {
                "date": str(date.date()),                # Convert date to string for JSON serialization
                "portfolio": float(port_value) if port_value is not None else None,  # Portfolio value
                "spy": float(spy_val) if spy_val is not None else None,             # SPY value
            }
        )

    # Store the performance comparison data in the investment summary
    ctx.deps.state.investment_summary["performanceData"] = performanceData

    # STEP 11: Generate investment summary message
    """
    Create a human-readable summary of the investment results.
    This message will be displayed to the user or logged.
    """
    
    # Start with funding status message
    if add_funds_needed:
        msg = "Some investments could not be made due to insufficient funds. Please add more funds to your wallet.\n"
        # Detail each instance where funds were insufficient
        for d, t, p, c in add_funds_dates:
            msg += (
                f"On {d}, not enough cash for {t}: price ${p:.2f}, available ${c:.2f}\n"
            )
    else:
        msg = "All investments were made successfully.\n"
    
    # Add portfolio performance summary
    msg += f"\nFinal portfolio value: ${total_value:.2f}\n"
    msg += "Returns by ticker (percent and $):\n"
    
    # Detail returns for each stock
    for ticker in tickers:
        percent = percent_return_per_stock[ticker]       # Percentage return
        abs_return = returns[ticker]                     # Absolute return in dollars
        msg += f"{ticker}: {percent:.2f}% (${abs_return:.2f})\n"

    # STEP 12: Prepare state changes for UI update
    # Note: Tool messaging is currently commented out
    # ctx.state.messages.append(
    #     ToolMessage(
    #         role="tool",
    #         id=str(uuid.uuid4()),
    #         content="The relevant details had been extracted",
    #         tool_call_id=ctx.state.messages[-1].tool_calls[0].id,
    #     )
    # )
    
    tool_log_id = str(uuid.uuid4())                      # Generate ID for tool logging
    changes = []                                         # List of state changes
    
    # Add starting log entry to state changes
    changes.append(
        JSONPatchOp(
            op="add",
            path="/tool_logs/-",
            value={
                "message": "Starting cash allocation and performance calculation...",
                "status": "in_progress",
                "id": tool_log_start_id,
            },
        )
    )
    
    # Add completion log entry for cash allocation
    changes.append(
        JSONPatchOp(
            op="add",
            path="/tool_logs/-",
            value={
                "message": "Cash allocation and performance calculation completed",
                "status": "completed",
                "id": tool_log_id,
            },
        )
    )
    
    # STEP 13: Return state changes to update the UI
    return [StateDeltaEvent(type=EventType.STATE_DELTA, delta=changes)]


# ============================================================================
# AGENT TOOLS - Insights Generation
# ============================================================================


@agent.tool
async def generate_insights(
    ctx: RunContext[StateDeps[AgentState]],
    bullInsights: list[bull_insights],
    bearInsights: list[bear_insights],
    tickers: list[str],
) -> list[StateDeltaEvent]:
    """
    Generates and stores investment insights for the portfolio stocks.
    This function processes both positive (bullish) and negative (bearish) insights
    and prepares them for frontend rendering.
    
    This tool should be called after allocate_cash so as to generate insights based on the 
    stock tickers present in ctx.deps.state.investment_summary. 
    
    Make sure that each insight is unique and not repeated. For each company stock in the list 
    provided, you should generate 2 positive insights and 2 negative insights. 
    This tool should be called only once after allocating cash. At that time itself, 
    insights for all stock tickers need to be generated.

    Args:
        ctx: The current run context containing the agent state
        bullInsights: List of positive/bullish insights about the stocks
        bearInsights: List of negative/bearish insights about the stocks
        tickers: List of stock ticker symbols these insights relate to

    Returns:
        list[StateDeltaEvent]: List containing state change events for the UI
    """
    
    # Debug print to monitor the insights being generated
    print(bullInsights, bearInsights, tickers)
    
    # Generate unique identifier for this tool call
    tool_call_start_id = str(uuid.uuid4())
    tool_call_id = str(uuid.uuid4())
    
    # STEP 2: Prepare data for frontend chart and table rendering
    """
    Package the investment summary and insights data together for the frontend.
    This combined data structure will be used to render charts, tables, and insights
    in the user interface.
    """
    ctx.deps.state.render_standard_charts_and_table_args = (
        {
            # Include the complete investment summary (portfolio performance, holdings, etc.)
            "investment_summary": ctx.deps.state.investment_summary,
            
            # Include the insights data, converting Pydantic models to dictionaries
            "insights": {
                "bullInsights": [insight.model_dump() for insight in bullInsights],  # Convert to dict
                "bearInsights": [insight.model_dump() for insight in bearInsights],   # Convert to dict
            },
        }
        # Note: default=str parameter would convert non-serializable objects to strings if needed
    )
    
    # STEP 3: Return state changes to update the UI
    """
    Return state changes including:
    1. Add the starting tool log entry
    2. Add the completion tool log entry  
    3. Update the render arguments for the frontend components
    """
    return [
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                # Add the starting tool log entry
                {
                    "op": "add",
                    "path": "/tool_logs/-",
                    "value": {
                        "message": "Starting insights generation...",
                        "status": "in_progress",
                        "id": tool_call_start_id,
                    },
                },
                
                # Add the completion tool log entry for this operation
                {
                    "op": "add",
                    "path": "/tool_logs/-",
                    "value": {
                        "message": "Investment insights generated successfully",
                        "status": "completed",
                        "id": tool_call_id,
                    },
                },
                
                # Update the rendering arguments for frontend components
                {
                    "op": "replace",
                    "path": "/render_standard_charts_and_table_args",
                    "value": ctx.deps.state.render_standard_charts_and_table_args,
                },
            ],
        ),
    ]


# ============================================================================
# AGENT EXPORT - Final Agent Configuration
# ============================================================================

# Convert the agent to AG-UI format with initial state dependencies
# This creates the final agent instance that will be used by the application
# pydantic_agent = agent.to_ag_ui(deps=StateDeps(AgentState()))