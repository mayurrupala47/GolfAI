import supervision as sv
import inspect

def main():
    print("--- ByteTrack Signature ---")
    sig = inspect.signature(sv.ByteTrack.__init__)
    print(sig)
    print("\n--- ByteTrack Docstring ---")
    print(sv.ByteTrack.__doc__)

if __name__ == "__main__":
    main()
