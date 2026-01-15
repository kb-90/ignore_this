# xrpscan.py
import aiohttp
import websockets
import json
import os
import asyncio
import time
import numpy as np
from typing import Set, Dict, List, Any
from terminal_styles import TerminalStyle

class IntelligentWhaleScanner:
    """True Omniscient whale detector – full ledger audit mirroring XRPScan Console"""
    
    def __init__(self):
        self.ws_url = "wss://xrplcluster.com"
        self.exchange_addresses: Set[str] = set()
        self.exchange_name_map: Dict[str, str] = {}  # address → name for pretty output
        self.whale_threshold = 250_000  # XRP
        self.hours_back = 6
        self.ledgers_back = 5400  # ~6 hours
        self.ledgers_per_batch = 20
        self.decay_lambda = 0.5  # Controls how fast old data "fades" (0.5 = moderate decay)

        # Updated heavy-hitter fallbacks (accurate as per your latest IDs)
        # Updated heavy-hitter fallbacks (full top 100 from latest list as of Jan 10, 2026)
        self.fallback_exchanges = {
            "Private Wallet - Massive Holders (Top 1-8)": [
                "rB3WNZc45gxzW31zxfXdkx8HusAhoqscPn",   # 5B+, +2345%
                "r9UUEXn3cx2seufBkDa8F86usfjWM6HiYp",   # 5B+, +2205%
                "rDdXiA3M4mYTQ4cFpWkVXfc2UaAXCFWeCK",   # 5B+, +2165%
                "rKDvgGUsNPZxsgmoemfrgXPS2Not4co2op",   # 5B+, +2015%
                "rMhkqz3DeU7GUUJKGZofusbrTwZe6bDyb1",   # 4.5B+, +2064.5%
                "r9NpyVfLfUG8hatuCCHKzosyDtKnBdsEN3",   # 4.5B+, +2044.5%
                "rN8pqRwLYuuvY7pUHurybPC8P6rLqVsu6o",   # 3.1B+, +2003.1%
                "rKwJaGmB5Hz24Qs2iyCaTdUuL1WsEXUWy5",   # 3B+, +2013%
            ],
            "Binance": [
                "rs8ZPbYqgecRcDzQpJYAMhSxSi5htsjnza",   # 1.705B+, +1.71%
            ],
            "Uphold": [
                "rsXT3AQqhHDusFs3nQQuwcA1yXRLZJAXKw",   # 1.514B+, +1.51%
            ],
            "Ripple Labs": [
                "rMQ98K56yXJbDGv49ZSmW51sLn94Xe1mu1",   # 1.33B+, +1.33%
            ],
            "Upbit": [
                "rDxJNbV23mu9xsWoQHoBqZQvc77YcbJXwb",   # 1.23B+, +1.23%
                "r4G689g4KePYLKkyyumM1iUppTP4nhZwVC",   # 500M+, +0.50%
                "rJo4m69u9Wd1F8fN2RbgAsJEF6a4hW1nSi",   # 500M+, +0.50%
                "rLgn612WAgRoZ285YmsQ4t7kb8Ui3csdoU",   # 500M+, +0.50%
                "rs48xReB6gjKtTnTfii93iwUhjhTJsW78B",   # 500M+, +0.50%
                "rJWbw1u3oDDRcYLFqiWFjhGWRKVcBAWdgp",   # 500M+, +0.50%
                "rMNUAfSz2spLEbaBwPnGtxTzZCajJifnzH",   # 500M+, +0.50%
                "r38a3PtqW3M7LRESgaR4dyHjg3AxAmiZCt",   # 500M+, +0.50%
            ],
            "Bitbank": [
                "rw7m3CtVHwGSdhFjV4MyJozmZJv3DYQnsA",   # 568M+, -37.7M change
            ],
            "Korbit": [
                "rsYFhEk4uFvwvvKJomHL7KhdF29r2sw9KD",   # 116M+, +0.12%
            ],
            "Private Wallet - Large Holders (9-20)": [
                "rLD5k36bJkNk1HkYSSCJwM4jBXChHjRViQ",   # 1.233B+, -71.18M
                "rKveEyR1SrkWbJX214xcfH43ZsoGMb3PEv",   # 845M+, +0.85%
                "rJ9Ey7HbscSECamgDRzvw5wrVbFUgaUDt7",   # 713M+, -9.31M
                "r99QSej32nAcjQAri65vE5ZXjw6xpUQ2Eh",   # 550M+, +0.55%
                "rBntsdo3fAS5sb3pqe7LvvxTS8qngFYAe1",   # 504M+, +0.50%
                "rw2hzLZgiQ9q62KCuaTWuFHWfiX7JWg3wY",   # 500M+, -5.62M
                "rGKHDyj4L6pc7DzRB6LWCR4YfZfzXj2Bdh",   # 500M+, +0.50%
                "rsjFB8mPWqiZgPUaVh8XYqdfa59PE2d5LG",   # 500M+, +0.50%
                "rDqGA2GfveHypDguQ1KXrJzYymFZmKxEsF",   # 500M+, +0.50%
                "rHGfmgv54kpc3QCZGRXEQKUhLPndbasbQr",   # 500M+, +0.50%
                "rp6aTJmW3nq1aKt3Jmuz4DPRxksT5PBjpH",   # 500M+, +0.50%
                "rfL1mn4VTCoHdhHhHMwqpShCFUaDBRk6Z5",   # 500M+, +0.50%
            ],
            "Private Wallet - Mid-Tier Holders (21-50)": [
                "rwa7YXssGVAL9yPKw6QJtCen2UqZbRQqpM",   # 500M+, +0.50%
                "rNcAdhSLXBrJ3aZUq22HaNtNEPpB5fR8Ri",   # 500M+, +0.50%
                "r4G689g4KePYLKkyyumM1iUppTP4nhZwVC",   # 500M+, +0.50% (Upbit duplicate)
                "rJo4m69u9Wd1F8fN2RbgAsJEF6a4hW1nSi",   # 500M+, +0.50% (Upbit duplicate)
                "rLgn612WAgRoZ285YmsQ4t7kb8Ui3csdoU",   # 500M+, +0.50% (Upbit duplicate)
                "rs48xReB6gjKtTnTfii93iwUhjhTJsW78B",   # 500M+, +0.50% (Upbit duplicate)
                "rJWbw1u3oDDRcYLFqiWFjhGWRKVcBAWdgp",   # 500M+, +0.50% (Upbit duplicate)
                "rMNUAfSz2spLEbaBwPnGtxTzZCajJifnzH",   # 500M+, +0.50% (Upbit duplicate)
                "r38a3PtqW3M7LRESgaR4dyHjg3AxAmiZCt",   # 500M+, +0.50% (Upbit duplicate)
                "rDfrrrBJZshSQDvfT2kmL9oUBdish52unH",   # 500M+, +0.50%
                "rD6tdgGHG7hwGTA6P39aE7W89fbqxXRjzk",   # 500M+, +0.50%
                "r476293LUcDqtjiSGJ5Dh44J1xBCDWeX3",   # 500M+, +0.50%
                "rEvwSpejhGTbdAXbxRTpGAzPBQkBRZxN5s",   # 461M+, +0.46%
                "r44CNwMWyJf4MEA1eHVMLPTkZ1LSv4Bzrv",   # 450M+, +0.45%
                "rH5wodHpZzeXBAWE36nMoRXGqeEjSdbzWU",   # 387M+, -10M
                "rpPcmcGQ5iTXDc5zF5owxwTifkTs1qYrA6",   # 384M+, -315k
                "rB1kVfLSxpXCw7sLCBcm5LFZYzkS6xmwSK",   # 333M+, +4.86M
                "rJpj1Mv21gJzsbsVnkp1U4nqchZbmZ9pM5",   # 325M+, +0.33%
                "rwshjBngGqMRJgGYvEJXGMkg5DS2GX3U3q",   # 317M+, +0.32%
                "rPoJNiCk7XSFLR28nH2hAbkYqjtMC3hK2k",   # 300M+, +0.30%
                "rDKw32dPXHfoeGoD3kVtm76ia1WbxYtU7D",   # 292M+, +5.44M
                "rhREXVHV938ToGkdJQ9NCYEY4x8kSEtjna",   # 282M+, +0.28%
                "rKNwXQh9GMjaU8uTqKLECsqyib47g5dMvo",   # 260M+, +138M
                "rstryhbE73v18SnJ3R8j1FSNYFWCSdELEd",   # 252M+, +7.46M
                "rJqiMb94hyz41SBTNr2AyPNW8AzELa8nE",   # 239M+, +0.24%
                "rfCKgAfaY2GaRFyCrwoF6BAhsEyLuWp37N",   # 236M+, +19.97M
                "rE5LDXksLZHsRgGUgqu7NiTSDd5zFz7rsW",   # 230M+, +0.23%
                "rprAu33H7PLUc24EYiMcD3HKcZG18PFkzQ",   # 212M+, -40M
                "rN1yT2hkfMt89CJVsXdvnKqRJbqm7TC8uo",   # 206M+, +1.82M
                "ragKXjY7cBTXUus32sYHZVfkY46Nt2Q829",   # 200M+, -29.27M
                "rG2eEaeiJou6cVQ3KtX7XMNwGhuW99xmHP",   # 200M+, +0.20%
            ],
            "Private Wallet - Smaller Significant Holders (51-100)": [
                "rsF9cc6gniHLTR2Jng29ng21ez7L9PpmPt",   # 200M+, +0.20%
                "rJ5EJYsW6Vkeruj1LAmQYq3VP7QUQKBH1W",   # 200M+, +0.20%
                "rsXNUCJkXeyFuGHyfRnuWPita2ns32upBD",   # 200M+, +0.20%
                "rQKZSMgmBJvv3FvWj1vuGjUXnegTqJc25z",   # 199M+, +0.20%
                "rhtufNsYfrozs4GvSq4HMYcR9y3dg8FWdC",   # 195M+, +0.20%
                "rhWVCsCXrkwTeLBg6DyDr7abDaHz3zAKmn",   # 183M+, +0.18%
                "rP8GfS4Ku43STM9kHEoKeWVvhV1E525zfo",   # 180M+, +0.18%
                "rMXWrmn3FpmA65UPyzDTez4Jt29NeqkKes",   # 169M+, +0.17%
                "rHJgQ4Cbg7vACGVuGusaKfmr2nheCRefBS",   # 163M+, +0.16%
                "r4AUYDBeV8YaLDZwuXG28CQgZ8XrThy8F2",   # 163M+, +0.16%
                "rH4nomQDy64MG5QGJngNS9cgGCdTFrGqLE",   # 160M+, +0.16%
                "rHjxBjzGcZKkPUwqrgaPYrk53PtLTXp23K",   # 160M+, +0.16%
                "rEq4b7nbL2ep44Fgk9bPwpynGRjyESpf5B",   # 159M+, +0.16%
                "rP3mUZyCDzZkTSd1VHoBbFt8HGm8fyq8qV",   # 158M+, +6.1M
                "rBEc94rUFfLfTDwwGN7rQGBHc883c2QHhx",   # 145M+, +0.15%
                "rLuQnupL8NQJX9Ywc59cxNrNPHsCHU1XK6",   # 141M+, +1.42M
                "rGMiNvZB2kcoXHv81BFRvaAkrSsiy9bQ9j",   # 138M+, -7M
                "rDecw8UhrZZUiaWc91e571b3TL41MUioh7",   # 133M+, +3.65M
                "rBB8peCvJcSbkmuQSe6ct6cujqNzz465cB",   # 131M+, +0.13%
                "rEbXa31msPbPDZgmLMKH7CaKaf7VipoLBo",   # 126M+, -4M
                "rJEvHUWgE5eb3R3p8cSaFHqh8Q2mUZqzsp",   # 126M+, +4.17M
                "rLUrobvcPHmbRVgzgGA6Vsp7Eu7yBQpEQe",   # 125M+, +5.16M
                "rEvuKRoEbZSbM5k5Qe5eTD9BixZXsfkxHf",   # 119M+, -40M
                "rsYFhEk4uFvwvvKJomHL7KhdF29r2sw9KD",   # 116M+, +0.12% (Korbit duplicate)
                "rH1dEntS4VgPBfzQTXZqTuWCAcq6g3xqLm",   # 112M+, +0.11%
                "rwTTsHVUDF8Ub2nzV2oAeWxfJzUvobXLEf",   # 112M+, +0.11%
                "rQf2ispc3QBvprB6S4fpiS1HbbSce9G4b8",   # 110M+, +4M
                "rBWEYyxPZkDPgBZEj73vgxi8xrNY22pnM7",   # 106M+, +0.11%
                "rE84wNj2fKZtiH3KCF77mZeUg5fypjPasw",   # 106M+, -2.46M
                "raLybBkX8HMsFG4EJGnTsBiNhnJS1Lqwmn",   # 106M+, -2.29M
                "rsyDbFZwxUqXEzwknqzCvYxk2davoQCUDC",   # 106M+, -0.38M
                "rG71mF18FKc6sWfCyfYiPHax1GwBNfqGFQ",   # 106M+, -0.63M
                "rLGbi542GmWboteyBAdBaRj65wBLkmDis9",   # 106M+, +1.30M
                "rsnnbMctkVJiXdV6aPMRPeaEu2uAnq3rEK",   # 103M+, +3.51M
                "rW1gMG9wKxEtcxtEYihSwVLqpvQ37Y6Sv",   # 103M+, +0.10%
                "rLjYpsikc5dPhGCt5f5FiUeKyCobX8eRSe",   # 103M+, +0.10%
                "rH1dGoeLbKbf2HNv22Ryhx9ATf87M1hQKA",   # 102M+, +0.10%
                "rGzwBVxutLLaxfeE4mJWrxHX1SMRxjo7Am",   # 102M+, +0.10%
                "rNYqPrNofcA6GZaK51i4TC7vgpjvJAECqx",   # 102M+, -23.45M
                "rQUwf3NHvHAahqgNqFuUs1gmk3zMUZ7U36",   # 101M+, +2.13k
                "rpNF4938Y8zCrqFP2owDHjMUdpAxMs49JD",   # 100M+, +0.10%
                "rB5AptBVH8nzavopD9PCbkPgj1DHSkau1w",   # 100M+, +0.10%
                "rLBHKfeHUf6RK3y3ij3y6Ea3b7MxqaLgK3",   # 100M+, +0.10%
            ]
        }

    def calculate_time_decay_weight(self, tx_time_seconds: float) -> float:
        """Calculates weight based on how recent the transaction is."""
        current_time = time.time()
        # Time difference in hours
        hours_ago = (current_time - tx_time_seconds) / 3600
        # Ensure we don't have negative time if clocks are slightly off
        hours_ago = max(0, hours_ago)
        # Exponential decay: e^(-lambda * hours)
        return np.exp(-self.decay_lambda * hours_ago)

    async def fetch_exchange_addresses_from_xrpscan(self):
        """Fetch well-known exchange addresses from the correct, working XRPSCAN endpoint"""
        TerminalStyle.info("Updating exchange address cache from XRPScan...")
        url = "https://api.xrpscan.com/api/v1/names/well-known"  # Confirmed working endpoint
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        data = json.loads(text)
                        for item in data:
                            account = item.get("account")
                            name = item.get("name", "Unknown Exchange")
                            if account:
                                self.exchange_addresses.add(account)
                                self.exchange_name_map[account] = name
                        TerminalStyle.success(f"Fetched {len(self.exchange_addresses)} labeled addresses from XRPSCAN")
                        return
                    else:
                        TerminalStyle.warning(f"XRPScan API returned {resp.status}. Falling back to hardcoded list.")
        except Exception as e:
            TerminalStyle.warning(f"XRPScan API failed ({e}). Falling back to hardcoded list.")

        # Robust fallback
        for name, addrs in self.fallback_exchanges.items():
            for addr in addrs:
                self.exchange_addresses.add(addr)
                self.exchange_name_map[addr] = name
        TerminalStyle.success(f"Loaded {len(self.exchange_addresses)} fallback exchange addresses")

    async def get_current_ledger(self, ws):
        request = {"command": "ledger_current"}
        await ws.send(json.dumps(request))
        response = await ws.recv()
        return json.loads(response).get("ledger_current_index", 0)

    async def perform_full_audit(self, print_output: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Full audit of ledgers to detect high-value movements.
        Returns a list of significant transactions with timestamps.
        """
        if print_output:
            TerminalStyle.subheader("OMNISCIENT WHALE AUDIT")
        
        # Ensure exchange addresses are loaded
        await self.fetch_exchange_addresses_from_xrpscan()
        
        raw_transactions = []
        
        async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=30, max_size=10_000_000) as websocket:
            # 1. Get current ledger height
            await websocket.send(json.dumps({"command": "ledger_closed"}))
            response = json.loads(await websocket.recv())
            latest_ledger = response['result']['ledger_index']
            
            # Use .env or default for threshold
            current_threshold = float(os.getenv("WHALE_THRESHOLD", self.whale_threshold))
            
            start_ledger = max(latest_ledger - self.ledgers_back, 1)
            
            if print_output:
                TerminalStyle.info(f"Scanning {latest_ledger - start_ledger + 1:,} ledgers (from {start_ledger}) for moves > {current_threshold:,.0f} XRP...")

            # 2. Batch Processing
            for batch_start in range(start_ledger, latest_ledger, self.ledgers_per_batch):
                batch_end = min(batch_start + self.ledgers_per_batch - 1, latest_ledger)
                
                tasks = []
                for idx in range(batch_start, batch_end + 1):
                    # Request ledger with transactions AND header data (for close_time)
                    tasks.append(websocket.send(json.dumps({
                        "command": "ledger",
                        "ledger_index": idx,
                        "transactions": True,
                        "expand": True
                    })))
                await asyncio.gather(*tasks)

                for _ in range(batch_start, batch_end + 1):
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    
                    ledger_result = data.get('result', {}).get('ledger', {})
                    transactions = ledger_result.get('transactions', [])
                    
                    # Extract ledger close time (Ripple Epoch -> Unix Timestamp)
                    # Ripple Epoch starts Jan 1, 2000 (946684800)
                    close_time_ripple = ledger_result.get('close_time')
                    if close_time_ripple is not None:
                        ledger_timestamp = close_time_ripple + 946684800
                    else:
                        ledger_timestamp = time.time() # Fallback if missing (rare)

                    if not transactions: continue

                    for tx in transactions:
                        # Only look at successful Payments
                        if tx.get('TransactionType') == "Payment" and tx.get('metaData', {}).get('TransactionResult') == "tesSUCCESS":
                            
                            amount_raw = tx.get('Amount')
                            # XRP is a string of drops. Issued currencies are dicts.
                            if isinstance(amount_raw, str):
                                try:
                                    amount_xrp = float(amount_raw) / 1_000_000
                                except ValueError:
                                    continue
                                
                                if amount_xrp >= current_threshold:
                                    source = tx.get('Account')
                                    dest = tx.get('Destination')
                                    
                                    source_name = self.exchange_name_map.get(source, "Private Wallet")
                                    dest_name = self.exchange_name_map.get(dest, "Private Wallet")
                                    
                                    # Identify direction based on Exchange Cache
                                    is_source_exchange = source in self.exchange_addresses
                                    is_dest_exchange = dest in self.exchange_addresses
                                    
                                    tx_type = "TRANSFER"
                                    if is_source_exchange and not is_dest_exchange:
                                        tx_type = "INFLOW" # Whale bought/withdrew from exchange? 
                                        # Wait, definition check:
                                        # Source = Exchange, Dest = Private -> OUTFLOW from Exchange (or Inflow to Whale Wallet)
                                        # Usually "Inflow" means INTO the exchange (Sell pressure). "Outflow" means OUT of exchange (HODL).
                                        # Let's align with typical crypto logic:
                                        # Exchange -> Wallet = OUTFLOW (Bullish)
                                        # Wallet -> Exchange = INFLOW (Bearish)
                                        tx_type = "OUTFLOW" 
                                        if print_output:
                                            TerminalStyle.success(f"Whale WITHDRAWAL (Bullish): {amount_xrp:,.0f} XRP from {source_name}")
                                    
                                    elif is_dest_exchange and not is_source_exchange:
                                        tx_type = "INFLOW"
                                        if print_output:
                                            TerminalStyle.warning(f"Whale DEPOSIT (Bearish): {amount_xrp:,.0f} XRP to {dest_name}")
                                    
                                    else:
                                        if print_output:
                                            TerminalStyle.info(f"Whale TRANSFER: {amount_xrp:,.0f} XRP (Private → Private)")

                                    raw_transactions.append({
                                        'timestamp': ledger_timestamp,
                                        'amount': amount_xrp,
                                        'type': tx_type,
                                        'source': source,
                                        'destination': dest
                                    })
        
        return {'transactions': raw_transactions}

    async def scan_and_quantify(self, print_output=True):
        """Scans the ledger and returns quantified, time-weighted features."""
        
        # 1. Get raw data
        audit_result = await self.perform_full_audit(print_output=print_output)
        transactions = audit_result['transactions']
        
        # 2. Initialize metrics
        weighted_inflow = 0.0
        weighted_outflow = 0.0
        
        total_buy_vol = 0.0  # Defined as Outflow (Exchange -> Wallet)
        total_sell_vol = 0.0 # Defined as Inflow (Wallet -> Exchange)
        transfer_vol = 0.0
        
        buy_count = 0
        sell_count = 0
        transfer_count = 0

        # 3. Apply Time Decay
        for tx in transactions:
            weight = self.calculate_time_decay_weight(tx['timestamp'])
            amount = tx['amount']
            
            if tx['type'] == 'INFLOW': # Wallet -> Exchange (Potential Sell)
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
        
        # Quantified "Clairvoyance" Score
        # Positive = Bullish (more outflow/withdrawals recently)
        # Negative = Bearish (more inflow/deposits recently)
        whale_pressure_score = weighted_outflow - weighted_inflow
        
        net_flow_raw = total_buy_vol - total_sell_vol

        if print_output:
            TerminalStyle.subheader("WHALE AUDIT COMPLETE")
            TerminalStyle.success(f"Detected: {buy_count} WITHDRAWALS (Buys) | {sell_count} DEPOSITS (Sells) | {transfer_count} TRANSFERS")
            TerminalStyle.success(f"Net Exchange Flow: {net_flow_raw/1e6:+,.2f}M XRP")
            TerminalStyle.success(f"Time-Weighted Pressure Score: {whale_pressure_score:,.2f}")

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
            'exchange_addresses_known': len(self.exchange_addresses)
        }

    def get_quantified_features(self, audit_results):
        """
        Translates raw audit results into weighted 'Pressure Scores' 
        for the Neural Network.
        """
        # Use the existing weighted_outflow and weighted_inflow from perform_full_audit
        # Positive = Bullish (Withdrawals/Accumulation), Negative = Bearish (Deposits/Sell Pressure)
        whale_pressure = audit_results.get('whale_pressure_score', 0)
        
        return {
            'whale_pressure_score': round(whale_pressure, 4),
            'whale_net_flow_raw': audit_results.get('whale_net_flow', 0),
            'whale_vol_6h': audit_results.get('whale_total_volume', 0)
        }
