#!/usr/bin/env python3
"""
Quick start: Test Prosodic ABX integration

Usage:
    python scripts/test_prosodic_abx.py

This script tests the new HuBERT features and DTW comparison
using the Prosodic ABX framework.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import librosa
from jp_speech_eval.ssl_features import HuBERTFeatureExtractor
from jp_speech_eval.feature_comparison import FeatureComparator
from jp_speech_eval.phoneme_confusion import PhonemeConfusionDetector


def test_hubert_extraction():
    """Test HuBERT feature extraction."""
    print("=" * 60)
    print("TEST 1: HuBERT Feature Extraction")
    print("=" * 60)
    
    # Load example audio
    print("\n1. Loading example audio...")
    try:
        audio, sr = librosa.load(librosa.ex("brahms"), sr=16000, mono=True, duration=3)
        print(f"   ✓ Loaded audio: {len(audio)} samples at {sr} Hz")
    except Exception as e:
        print(f"   ✗ Failed to load audio: {e}")
        return False
    
    # Extract HuBERT features
    print("\n2. Initializing HuBERT-Large-Japanese...")
    try:
        extractor = HuBERTFeatureExtractor(
            model_id="facebook/hubert-large-ls60-japanese"
        )
        print(f"   ✓ Model loaded ({extractor.num_layers} layers)")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        print("   ℹ Make sure torch and transformers are installed:")
        print("     pip install torch transformers")
        return False
    
    # Extract features from all layers
    print("\n3. Extracting features from all layers...")
    try:
        features = extractor.extract_all_layers(audio, sr)
        print(f"   ✓ Extracted {len(features)} layers")
        
        # Show sample statistics
        for layer_idx in [0, 9, 15, 23]:
            if layer_idx in features:
                feat = features[layer_idx]
                print(f"     Layer {layer_idx:2d}: shape {feat.shape}, mean={np.mean(feat):.4f}, std={np.std(feat):.4f}")
    except Exception as e:
        print(f"   ✗ Failed to extract features: {e}")
        return False
    
    print("\n✅ HuBERT extraction test passed!")
    return True


def test_feature_comparison():
    """Test MFCC vs HuBERT DTW comparison."""
    print("\n" + "=" * 60)
    print("TEST 2: MFCC vs HuBERT DTW Comparison")
    print("=" * 60)
    
    # Load two audio segments
    print("\n1. Loading two audio segments...")
    try:
        audio1, sr = librosa.load(
            librosa.ex("brahms"), sr=16000, mono=True, duration=2, offset=0
        )
        audio2, _ = librosa.load(
            librosa.ex("brahms"), sr=sr, mono=True, duration=2, offset=2
        )
        print(f"   ✓ Loaded: audio1={len(audio1)} samples, audio2={len(audio2)} samples")
    except Exception as e:
        print(f"   ✗ Failed to load audio: {e}")
        return False
    
    # Compare representations
    print("\n2. Comparing representations (MFCC + HuBERT layers)...")
    try:
        comparator = FeatureComparator()
        results = comparator.compare_representations(
            audio1, audio2, sr=sr,
            hubert_layers=[9, 15, 23]  # Key layers from Prosodic ABX paper
        )
        
        print("   DTW Costs:")
        for name, cost in sorted(results.items()):
            if "error" not in name:
                print(f"     {name:20s}: {cost:.4f}")
            else:
                print(f"     {name:20s}: {cost}")
        
        print("\n   📊 Analysis:")
        mfcc_cost = results.get("mfcc", float("inf"))
        hubert_costs = [v for k, v in results.items() if k.startswith("hubert_layer_") and "error" not in k]
        
        if hubert_costs:
            best_layer = min(enumerate(hubert_costs), key=lambda x: x[1])
            print(f"     Best layer: {best_layer[0]} with cost {best_layer[1]:.4f}")
            
            if mfcc_cost != float("inf"):
                improvement = (mfcc_cost - best_layer[1]) / mfcc_cost * 100
                print(f"     Improvement over MFCC: {improvement:+.1f}%")
    except Exception as e:
        print(f"   ✗ Failed to compare: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ Feature comparison test passed!")
    return True


def test_confusion_detection():
    """Test phoneme confusion detection."""
    print("\n" + "=" * 60)
    print("TEST 3: Phoneme Confusion Detection")
    print("=" * 60)
    
    print("\n1. Creating simulated posteriorgrams...")
    np.random.seed(42)
    
    # Simulate confusion between phonemes 0 and 1
    n_frames = 100
    n_phonemes = 5
    phoneme_list = ["a", "i", "u", "e", "o"]
    
    posteriorgrams = np.random.dirichlet(np.ones(n_phonemes), size=n_frames)
    # Introduce confusion: make phonemes 0 and 1 similar
    posteriorgrams[:, 0] = 0.4 + 0.1 * np.random.randn(n_frames)
    posteriorgrams[:, 1] = 0.35 + 0.1 * np.random.randn(n_frames)
    posteriorgrams = np.abs(posteriorgrams) / np.sum(posteriorgrams, axis=1, keepdims=True)
    
    print(f"   ✓ Created posteriorgrams: shape {posteriorgrams.shape}")
    
    print("\n2. Detecting confused phoneme pairs...")
    try:
        detector = PhonemeConfusionDetector()
        
        # Detect with Bhattacharyya coefficient
        confused_bc = detector.detect_confusions(
            posteriorgrams,
            phoneme_list,
            similarity_metric="bc",
            threshold=0.3,
        )
        
        # Detect with KL divergence
        confused_kl = detector.detect_confusions(
            posteriorgrams,
            phoneme_list,
            similarity_metric="kl",
            threshold=0.5,
        )
        
        print(f"\n   Bhattacharyya Coefficient results:")
        for (p1, p2), score in confused_bc[:3]:
            print(f"     {p1} <-> {p2}: BC={score:.4f}")
        
        print(f"\n   KL Divergence results:")
        for (p1, p2), score in confused_kl[:3]:
            print(f"     {p1} <-> {p2}: KL={score:.4f}")
        
        print(f"\n   ✓ Found {len(confused_bc)} confused pairs (BC) and {len(confused_kl)} pairs (KL)")
    except Exception as e:
        print(f"   ✗ Failed to detect confusions: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ Phoneme confusion detection test passed!")
    return True


def main():
    print("\n" + "=" * 60)
    print("Prosodic ABX Integration Tests")
    print("Testing framework from Sun & McIntosh (INTERSPEECH 2026)")
    print("=" * 60)
    
    results = []
    
    # Test 1: HuBERT extraction
    try:
        results.append(("HuBERT Extraction", test_hubert_extraction()))
    except Exception as e:
        print(f"\n❌ Test 1 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("HuBERT Extraction", False))
    
    # Test 2: Feature comparison
    try:
        results.append(("Feature Comparison", test_feature_comparison()))
    except Exception as e:
        print(f"\n❌ Test 2 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Feature Comparison", False))
    
    # Test 3: Confusion detection
    try:
        results.append(("Confusion Detection", test_confusion_detection()))
    except Exception as e:
        print(f"\n❌ Test 3 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Confusion Detection", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
