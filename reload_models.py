#!/usr/bin/env python3
"""
Script to reload models in a running miner without restarting.
This connects to the running miner process and triggers model reload.

Usage:
    python reload_models.py
"""

import sys
import os
import signal
import subprocess

def find_miner_process():
    """Find the running miner process."""
    try:
        result = subprocess.run(
            ['pm2', 'jlist'],
            capture_output=True,
            text=True,
            check=True
        )
        import json
        processes = json.loads(result.stdout)
        for proc in processes:
            if proc.get('name') == 'zeus_miner':
                return proc.get('pid')
    except Exception as e:
        print(f"Error finding miner process: {e}")
    return None

def reload_via_signal():
    """Send USR1 signal to trigger reload (if signal handler is set up)."""
    pid = find_miner_process()
    if pid:
        try:
            os.kill(pid, signal.SIGUSR1)
            print(f"Sent reload signal to miner process {pid}")
            return True
        except Exception as e:
            print(f"Error sending signal: {e}")
    else:
        print("Miner process not found")
    return False

if __name__ == "__main__":
    print("=" * 80)
    print("MODEL RELOAD HELPER")
    print("=" * 80)
    
    # Find miner process
    pid = find_miner_process()
    
    if pid:
        print(f"\nFound miner process: PID {pid}")
        print("\nSending reload signal (SIGUSR1)...")
        
        if reload_via_signal():
            print("✓ Signal sent successfully!")
            print("\nCheck miner logs to confirm models were reloaded:")
            print("  pm2 logs zeus_miner --lines 50")
        else:
            print("\nFailed to send signal. You can restart the miner instead:")
            print("  pm2 restart zeus_miner")
    else:
        print("\n❌ Miner process not found. Is it running?")
        print("\nTo start the miner:")
        print("  ./start_miner.sh")
    
    print("\n" + "=" * 80)
    print("Alternative: Restart miner (simpler, but causes brief downtime):")
    print("  pm2 restart zeus_miner")
    print("=" * 80)

