"""
Simple script to test SUMO and traci installation
"""

import os
import sys
import subprocess
import platform

def check_sumo_home():
    """Check if SUMO_HOME environment variable is set"""
    sumo_home = os.environ.get('SUMO_HOME')
    if sumo_home:
        print(f"✅ SUMO_HOME is set to: {sumo_home}")
        if os.path.exists(sumo_home):
            print(f"✅ SUMO_HOME directory exists")
        else:
            print(f"❌ SUMO_HOME directory does not exist: {sumo_home}")
            return False
    else:
        print("❌ SUMO_HOME environment variable is not set")
        return False
    
    return True

def check_sumo_binaries():
    """Check if SUMO binaries are in PATH"""
    try:
        if platform.system() == 'Windows':
            sumo_binary = 'sumo.exe'
            sumo_gui_binary = 'sumo-gui.exe'
        else:
            sumo_binary = 'sumo'
            sumo_gui_binary = 'sumo-gui'
        
        # Check if binaries are in PATH
        result = subprocess.run(['where' if platform.system() == 'Windows' else 'which', sumo_binary], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {sumo_binary} found in PATH: {result.stdout.strip()}")
        else:
            # Check if binaries are in SUMO_HOME/bin
            sumo_home = os.environ.get('SUMO_HOME')
            if sumo_home:
                sumo_path = os.path.join(sumo_home, 'bin', sumo_binary)
                if os.path.exists(sumo_path):
                    print(f"✅ {sumo_binary} found in SUMO_HOME/bin")
                else:
                    print(f"❌ {sumo_binary} not found in PATH or SUMO_HOME/bin")
                    return False
            else:
                print(f"❌ {sumo_binary} not found in PATH and SUMO_HOME not set")
                return False
    except Exception as e:
        print(f"❌ Error checking SUMO binaries: {e}")
        return False
    
    return True

def check_traci_module():
    """Check if traci module can be imported"""
    print("\nTrying to import traci module...")
    
    # First, check if traci is directly importable
    try:
        import traci
        print("✅ Successfully imported traci module!")
        return True
    except ImportError:
        print("❌ Could not import traci directly")
    
    # Try importing after adding SUMO_HOME/tools to path
    sumo_home = os.environ.get('SUMO_HOME')
    if sumo_home:
        tools_path = os.path.join(sumo_home, 'tools')
        print(f"Adding {tools_path} to Python path and trying again...")
        sys.path.append(tools_path)
        
        try:
            import traci
            print("✅ Successfully imported traci after adding SUMO_HOME/tools to path!")
            print("NOTE: You'll need to add this path in your scripts:")
            print(f"import os, sys; sys.path.append('{tools_path}')")
            return True
        except ImportError:
            print("❌ Still could not import traci")
    
    return False

def main():
    """Main function"""
    print("=" * 50)
    print("SUMO and TraCI Installation Test")
    print("=" * 50)
    
    sumo_home_ok = check_sumo_home()
    binaries_ok = check_sumo_binaries()
    traci_ok = check_traci_module()
    
    print("\n" + "=" * 50)
    if sumo_home_ok and binaries_ok and traci_ok:
        print("✅ SUMO and TraCI are correctly installed and configured!")
        print("You can run the Traffic Signal Management System.")
    else:
        print("❌ There are issues with your SUMO installation.")
        print("\nTo fix this:")
        print("1. Install SUMO from https://sumo.dlr.de/docs/Downloads.php")
        print("2. Set the SUMO_HOME environment variable")
        print("3. Add SUMO/bin to your PATH environment variable")
        print("4. Ensure SUMO/tools is in your Python path")
    print("=" * 50)

if __name__ == "__main__":
    main()
