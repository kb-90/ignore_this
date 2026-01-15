import re
import os
import sys

class TerminalColors:
    """ANSI color codes for terminal styling"""
    BLUE = '\x1b[96m'        # Blue for info
    GREEN = '\x1b[92m'       # Green for success
    ORANGE = '\x1b[93m'      # Orange for warnings
    RED = '\x1b[91m'         # Red for errors
    ENDC = '\x1b[0m'         # Reset
    BOLD = '\x1b[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'
    # Custom colors for cyberpunk theme: Purple, Pink, Turquoise, with Orange accent
    TURQUOISE = '\033[38;5;51m'
    PURPLE = '\033[38;5;141m'
    PINK = '\033[38;5;213m'

class TerminalStyle:
    """Enhanced terminal output with modern, trader-friendly design"""
    
    width = 80

    @staticmethod
    def header(title, subtitle1, subtitle2, subtitle3, author):
        """Main section header with gradient effect"""
        print(f"\n{TerminalColors.BOLD}{TerminalColors.TURQUOISE}╭{'─' * (TerminalStyle.width-2)}╮{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}│{TerminalColors.PINK}{title.center(TerminalStyle.width-2)}{TerminalColors.TURQUOISE}│{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}│{'─' * (TerminalStyle.width-2)}│")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}│{TerminalColors.PURPLE}{subtitle1.center(TerminalStyle.width-2)}{TerminalColors.TURQUOISE}│{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}│{TerminalColors.PURPLE}{subtitle2.center(TerminalStyle.width-2)}{TerminalColors.TURQUOISE}│{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}│{TerminalColors.PURPLE}{subtitle3.center(TerminalStyle.width-2)}{TerminalColors.TURQUOISE}│{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}│{'─' * (TerminalStyle.width-2)}│")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}│{TerminalColors.BLUE}{author.center(TerminalStyle.width-2)}{TerminalColors.TURQUOISE}│{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}╰{'─' * (TerminalStyle.width-2)}╯{TerminalColors.ENDC}")

    @staticmethod
    def subheader(text):
        """Subsection header"""
        print(f"{TerminalColors.TURQUOISE}╭{'─' * (TerminalStyle.width-2)}╯{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.PINK}▶ {text.upper()}{TerminalColors.ENDC}")
        print(f"{TerminalColors.TURQUOISE}╰{'─' * (TerminalStyle.width - 2)}╮{TerminalColors.ENDC}")
    
    @staticmethod
    def section_open(text):
        """Prints an opening section header."""
        print(f"  {TerminalColors.TURQUOISE}╭────╮{' ' * (TerminalStyle.width - 9)}│{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}╭─│ {TerminalColors.PURPLE}{text.upper()} {TerminalColors.TURQUOISE}{'─' * (TerminalStyle.width - len(text) - 6)}╯{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}│ ╰────╯{TerminalColors.ENDC}")

    @staticmethod
    def training_pipeline_subheader(text):
        """Prints a training pipeline subheader."""
        print(f"{TerminalColors.TURQUOISE}╭{'─' * (TerminalStyle.width-2)}╯{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.PINK}✦ {text.upper()}{TerminalColors.ENDC}")
        print(f"{TerminalColors.TURQUOISE}╰{'─' * (TerminalStyle.width - 2)}╮{TerminalColors.ENDC}")

    @staticmethod
    def training_subheader(text):
        """Prints a training subheader."""
        print(f"  {TerminalColors.TURQUOISE}╭────╮{' ' * (TerminalStyle.width - 9)}│{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}╭─│‧˚✦ : {TerminalColors.PURPLE}{text.upper()} {TerminalColors.TURQUOISE}{'─' * (TerminalStyle.width - len(text) - 11)}╯{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}│ ╰────╯{TerminalColors.ENDC}")

    @staticmethod
    def section_close():
        """Prints a closing section line."""
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}╰{'─' * (TerminalStyle.width-2)}╮{TerminalColors.ENDC}")

    @staticmethod
    def blank_section():
        """Prints an opening section line without text."""
        print(f"{TerminalColors.TURQUOISE}╭{'─' * (TerminalStyle.width-2)}╯{TerminalColors.ENDC}")

    @staticmethod
    def line_break():
        """Prints a line break with the continuous line."""
        print(f"{TerminalColors.TURQUOISE}│{' ' * (TerminalStyle.width-1)}{TerminalColors.ENDC}")

    @staticmethod
    def success(text):
        """Success message"""
        print(f"  {TerminalColors.GREEN}✓ {text}{TerminalColors.ENDC}")
    
    @staticmethod
    def info(text):
        """Info message"""
        print(f"  {TerminalColors.BLUE}● {text}{TerminalColors.ENDC}")
    
    @staticmethod
    def warning(text):
        """Warning message"""
        print(f"  {TerminalColors.ORANGE}⚠ {text}{TerminalColors.ENDC}")
    
    @staticmethod
    def error(text):
        """Error message"""
        print(f"  {TerminalColors.RED}✗ {text}{TerminalColors.ENDC}")
    
    @staticmethod
    def metric(label, value, unit="", positive=None):
        """Display a metric with optional color coding"""
        color = TerminalColors.ENDC
        if positive is True:
            color = TerminalColors.GREEN
        elif positive is False:
            color = TerminalColors.RED
        
        print(f"  ⓘ {TerminalColors.DIM}{label}:{TerminalColors.ENDC} {color}{TerminalColors.BOLD}{value}{unit}{TerminalColors.ENDC}")

    @staticmethod
    def prediction_box(current, predicted, change_pct, horizon, confidence_lower=None, confidence_upper=None, confidence_score=None):
        """Enhanced prediction display with visual indicators"""
        color = TerminalColors.GREEN if change_pct > 0 else TerminalColors.RED
        symbol = "▲" if change_pct > 0 else "▼"
        width = 58

        def strip_ansi(text):
            return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)

        conf_color = TerminalColors.ENDC
        conf_bar = ""
        if confidence_score is not None:
            bar_length = int(confidence_score / 5)
            if confidence_score >= 75:
                conf_color = TerminalColors.GREEN
                conf_bar = "█" * bar_length
            elif confidence_score >= 50:
                conf_color = TerminalColors.ORANGE
                conf_bar = "▓" * bar_length
            else:
                conf_color = TerminalColors.RED
                conf_bar = "░" * bar_length
        
        def print_line(content):
            padding = width - len(strip_ansi(content))
            print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}║{TerminalColors.ENDC}{content}{' ' * padding}{TerminalColors.BOLD}{TerminalColors.TURQUOISE}║{TerminalColors.ENDC}")

        print(f"\n{TerminalColors.BOLD}{TerminalColors.TURQUOISE}╔{'═' * width}╗{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}║{TerminalColors.PINK}{f'  {horizon}H FORECAST'.center(width)}{TerminalColors.TURQUOISE}║{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}╠{'═' * width}╣{TerminalColors.ENDC}")
        
        print_line(f"  Current Price:   {TerminalColors.BOLD}{current:>12.4f}{TerminalColors.ENDC} USDT")
        print_line(f"  Predicted Price: {TerminalColors.BOLD}{predicted:>12.4f}{TerminalColors.ENDC} USDT")
        
        if confidence_lower is not None and confidence_upper is not None:
            print_line(f"  Range:           {TerminalColors.DIM}{confidence_lower:.4f} - {confidence_upper:.4f}{TerminalColors.ENDC}")
        
        print_line(f"  Expected Change: {color}{TerminalColors.BOLD}{symbol} {abs(change_pct):>10.2f}%{TerminalColors.ENDC}")
        
        if confidence_score is not None:
            print_line(f"  Confidence:      {conf_color}{conf_bar} {TerminalColors.BOLD}{confidence_score:>5.1f}%{TerminalColors.ENDC}")
        
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}╚{'═' * width}╝{TerminalColors.ENDC}")

    @staticmethod
    def progress(current, total, task):
        """Enhanced progress indicator"""
        percentage = (current / total) * 100
        bar_length = 40
        filled = int(bar_length * current / total)
        bar = f"{TerminalColors.PINK}{'█' * filled}{TerminalColors.DIM}{'░' * (bar_length - filled)}{TerminalColors.ENDC}"
        print(f"\r  {TerminalColors.TURQUOISE}✇ {task}: {bar} {TerminalColors.BOLD}{percentage:.0f}%{TerminalColors.ENDC}", end='', flush=True)
        if current == total:
            print()

TerminalStyle = TerminalStyle()
