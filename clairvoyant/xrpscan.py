# xrpscan.py - High-Concurrency Architecture
import aiohttp
import websockets
import json
import os
import asyncio
import time
import numpy as np
from typing import Set, Dict, List, Any, Optional

from terminal_styles import TerminalStyle

class IntelligentWhaleScanner:
    """
    High-Concurrency, Time-Weighted Whale Scanner for XRPL.
    Uses a producer-consumer pattern to decouple network I/O from CPU-bound parsing.
    """
    
    def __init__(self, whale_threshold=250_000, hours_back=6, decay_lambda=0.5):
        self.ws_url = "wss://xrplcluster.com/"
        self.api_url = "https://api.xrpscan.com/api/v1/names"
        
        self.exchange_addresses: Set[str] = set()
        self.exchange_name_map: Dict[str, str] = {}
        
        self.whale_threshold = float(os.getenv("WHALE_THRESHOLD", whale_threshold))
        self.ledgers_back = int(hours_back * 3600 / 4)  # Approx. 4s per ledger
        self.decay_lambda = decay_lambda
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialized = False

    async def _initialize(self):
        """Initializes the session and fetches exchange labels."""
        if self._initialized:
            return
        
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        
        await self._fetch_exchange_labels()
        self._initialized = True

    async def close_session(self):
        """Closes the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _fetch_exchange_labels(self):
        """Caches exchange labels from XRPSCAN API at startup."""
        TerminalStyle.info("Updating exchange address cache from XRPScan...")
        try:
            assert self.session is not None
            async with self.session.get(self.api_url, timeout=15) as response:
                response.raise_for_status()
                data = await response.json()
                for item in data:
                    account = item.get("account")
                    name = item.get("name", "Unknown Exchange")
                    if account:
                        self.exchange_addresses.add(account)
                        self.exchange_name_map[account] = name
                TerminalStyle.success(f"Successfully cached {len(self.exchange_addresses)} exchange labels.")
        except (aiohttp.ClientError, AssertionError) as e:
            TerminalStyle.warning(f"API call to {self.api_url} failed: {e}. Exchange address resolution will be impaired.")
        except Exception as e:
            TerminalStyle.error(f"An unexpected error occurred during exchange label fetching: {e}")

    async def _producer(self, queue: asyncio.Queue, start_ledger: int, end_ledger: int):
        """Producer: Fetches raw ledger data and puts it into the queue."""
        retries = 3
        delay = 1.0
        
        for i in range(retries):
            try:
                # Set aggressive timeouts and size limits as requested
                async with websockets.connect(
                    self.ws_url, max_size=50_000_000, ping_interval=10, ping_timeout=20
                ) as ws:
                    for ledger_index in range(start_ledger, end_ledger + 1):
                        request = json.dumps({
                            "command": "ledger",
                            "ledger_index": ledger_index,
                            "transactions": True,
                            "expand": True
                        })
                        await ws.send(request)
                        response = await ws.recv()
                        await queue.put(json.loads(response))
                    return  # Success
            except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidStatusCode) as e:
                TerminalStyle.warning(f"WebSocket connection error: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            except Exception as e:
                TerminalStyle.error(f"Producer failed with unexpected error: {e}")
                break
        
        TerminalStyle.error("Producer failed after multiple retries. Halting scan.")
        await queue.put(None) # Signal consumer to stop even on failure

    async def _consumer(self, queue: asyncio.Queue, results_list: list, print_output: bool):
        """Consumer: Pulls from queue, parses transactions, and builds results."""
        while True:
            item = await queue.get()
            if item is None:  # Sentinel
                queue.task_done()
                break

            ledger_result = item.get('result', {}).get('ledger', {})
            if not ledger_result:
                queue.task_done()
                continue
            
            transactions = ledger_result.get('transactions', [])
            close_time_ripple = ledger_result.get('close_time', 0)
            ledger_timestamp = close_time_ripple + 946684800 # Ripple Epoch -> Unix
            
            for tx in transactions:
                if tx.get('TransactionType') == "Payment" and tx.get('metaData', {}).get('TransactionResult') == "tesSUCCESS":
                    amount_raw = tx.get('Amount')
                    if not isinstance(amount_raw, str): continue

                    try:
                        amount_xrp = float(amount_raw) / 1_000_000
                    except ValueError:
                        continue
                        
                    if amount_xrp >= self.whale_threshold:
                        source, dest = tx.get('Account'), tx.get('Destination')
                        is_source_exchange = source in self.exchange_addresses
                        is_dest_exchange = dest in self.exchange_addresses
                        
                        tx_type = "TRANSFER" # Default
                        if is_source_exchange and not is_dest_exchange:
                            tx_type = "OUTFLOW" # Exchange -> Wallet (Bullish accumulation)
                        elif not is_source_exchange and is_dest_exchange:
                            tx_type = "INFLOW" # Wallet -> Exchange (Bearish sell pressure)
                        
                        results_list.append({
                            'timestamp': ledger_timestamp, 'amount': amount_xrp, 'type': tx_type,
                            'source': source, 'destination': dest
                        })

            queue.task_done()

    async def perform_full_audit(self, print_output: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Main orchestration method for performing a high-concurrency ledger scan.
        This replaces the old sequential logic.
        """
        await self._initialize()
        
        if print_output:
            TerminalStyle.subheader("OMNISCIENT WHALE AUDIT (High-Concurrency)")

        start_ledger = 0
        latest_ledger = 0

        # 1. Get dynamic ledger index
        try:
            async with websockets.connect(self.ws_url, ping_interval=10) as ws:
                await ws.send(json.dumps({"command": "ledger_closed"}))
                response = json.loads(await ws.recv())
                latest_ledger = response['result']['ledger_index']
                start_ledger = latest_ledger - self.ledgers_back
        except Exception as e:
            TerminalStyle.error(f"Could not fetch latest ledger index: {e}. Aborting scan.")
            return {'transactions': []}
            
        if print_output:
            TerminalStyle.info(f"Scanning {self.ledgers_back:,} ledgers (from #{start_ledger}) for moves > {self.whale_threshold:,.0f} XRP...")

        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        results: List[Dict[str, Any]] = []

        # 2. Run Producer and Consumer concurrently
        producer_task = asyncio.create_task(self._producer(queue, start_ledger, latest_ledger))
        consumer_task = asyncio.create_task(self._consumer(queue, results, print_output))

        # 3. Wait for producer to finish, then for queue to be empty
        await producer_task
        await queue.join()
        
        # 4. Cleanly terminate consumer
        await queue.put(None) # Send sentinel
        await consumer_task
        
        if print_output:
            TerminalStyle.info(f"Scan complete. Found {len(results)} whale-sized transactions.")

        return {'transactions': results}

    def calculate_time_decay_weight(self, tx_time_seconds: float) -> float:
        """Calculates weight based on how recent the transaction is."""
        hours_ago = max(0, (time.time() - tx_time_seconds) / 3600)
        return np.exp(-self.decay_lambda * hours_ago)

    async def scan_and_quantify(self, print_output=True):
        """
        Scans the ledger using the new architecture and returns quantified, 
        time-weighted features for the neural network.
        """
        audit_result = await self.perform_full_audit(print_output=print_output)
        transactions = audit_result.get('transactions', [])
        
        weighted_inflow, weighted_outflow = 0.0, 0.0
        total_buy_vol, total_sell_vol, transfer_vol = 0.0, 0.0, 0.0
        buy_count, sell_count, transfer_count = 0, 0, 0

        for tx in transactions:
            weight = self.calculate_time_decay_weight(tx['timestamp'])
            amount = tx['amount']
            
            if tx['type'] == 'INFLOW': # Wallet -> Exchange (Sell Pressure)
                weighted_inflow += (amount * weight)
                total_sell_vol += amount
                sell_count += 1
            elif tx['type'] == 'OUTFLOW': # Exchange -> Wallet (Accumulation)
                weighted_outflow += (amount * weight)
                total_buy_vol += amount
                buy_count += 1
            else:
                transfer_vol += amount
                transfer_count += 1
        
        whale_pressure_score = weighted_outflow - weighted_inflow
        net_flow_raw = total_buy_vol - total_sell_vol

        if print_output:
            TerminalStyle.subheader("WHALE AUDIT QUANTIFIED")
            TerminalStyle.success(f"Detected: {buy_count} WITHDRAWALS | {sell_count} DEPOSITS | {transfer_count} TRANSFERS")
            TerminalStyle.success(f"Net Raw Exchange Flow: {net_flow_raw/1e6:+,.2f}M XRP")
            TerminalStyle.info(f"Time-Weighted Pressure Score: {whale_pressure_score:,.2f}")

        # Final dictionary for the ML model
        return {
            'whale_pressure_score': whale_pressure_score,
            'weighted_inflow': weighted_inflow,
            'weighted_outflow': weighted_outflow,
            'whale_buy_volume': total_buy_vol,
            'whale_sell_volume': total_sell_vol,
            'whale_net_flow': net_flow_raw,
            'whale_buy_count': buy_count,
            'whale_sell_count': sell_count,
            'whale_transfer_count': transfer_count,
            'whale_total_volume': total_buy_vol + total_sell_vol + transfer_vol,
        }

    def get_quantified_features(self, audit_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Translates the final quantified results into a feature set 
        for the neural network.
        """
        return {
            'whale_pressure_score': round(audit_results.get('whale_pressure_score', 0), 4),
            'whale_net_flow_raw': audit_results.get('whale_net_flow', 0),
            'whale_vol_6h': audit_results.get('whale_total_volume', 0)
        }

    async def __aenter__(self):
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_session()