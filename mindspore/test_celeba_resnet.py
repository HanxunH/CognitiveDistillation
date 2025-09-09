"""Test script for CelebA ResNet models in MindSpore"""
import mindspore as ms
import mindspore.numpy as mnp
from models.celeba_resnet import AttributesResNet18, AttributesResNet, BasicBlock, Bottleneck


def test_basic_block():
    """Test BasicBlock"""
    print("Testing BasicBlock...")
    block = BasicBlock(64, 64)
    x = ms.Tensor(mnp.ones((2, 64, 56, 56)), ms.float32)
    out = block(x)
    assert out.shape == (2, 64, 56, 56), f"Expected shape (2, 64, 56, 56), got {out.shape}"
    print("  BasicBlock test passed")


def test_bottleneck():
    """Test Bottleneck block"""
    print("Testing Bottleneck...")
    block = Bottleneck(256, 64)
    x = ms.Tensor(mnp.ones((2, 256, 56, 56)), ms.float32)
    out = block(x)
    assert out.shape == (2, 256, 56, 56), f"Expected shape (2, 256, 56, 56), got {out.shape}"
    print("  Bottleneck test passed")


def test_attributes_resnet18():
    """Test AttributesResNet18 model"""
    print("Testing AttributesResNet18...")
    
    # Test with default num_attributes
    model = AttributesResNet18(num_attributes=40)
    x = ms.Tensor(mnp.ones((2, 3, 224, 224)), ms.float32)
    
    # Test normal forward pass (multiple classifiers)
    outputs = model(x)
    assert len(outputs) == 40, f"Expected 40 outputs, got {len(outputs)}"
    assert all(out.shape == (2, 2) for out in outputs), "All outputs should have shape (2, 2)"
    print("  Multi-classifier forward pass test passed")
    
    # Test single classifier forward pass
    model.forward_idx = 10
    single_output = model(x)
    assert single_output.shape == (2, 2), f"Expected shape (2, 2), got {single_output.shape}"
    model.forward_idx = None
    print("  Single classifier forward pass test passed")
    
    # Test forward_fc method
    fc_output = model.forward_fc(x, 5)
    assert fc_output.shape == (2, 2), f"Expected shape (2, 2), got {fc_output.shape}"
    print("  forward_fc method test passed")
    
    # Test feature extraction mode
    model.get_features = True
    features = model(x)
    assert features.shape == (2, 512), f"Expected shape (2, 512), got {features.shape}"
    print("  Feature extraction mode test passed")
    
    # Test with different num_attributes
    model_10 = AttributesResNet18(num_attributes=10)
    outputs_10 = model_10(x)
    assert len(outputs_10) == 10, f"Expected 10 outputs, got {len(outputs_10)}"
    print("  Custom num_attributes test passed")


def test_attributes_resnet_custom():
    """Test custom AttributesResNet configurations"""
    print("Testing custom AttributesResNet configurations...")
    
    # Test with Bottleneck blocks (ResNet50-like architecture)
    model = AttributesResNet(Bottleneck, [3, 4, 6, 3], num_attributes=20)
    x = ms.Tensor(mnp.ones((1, 3, 224, 224)), ms.float32)
    outputs = model(x)
    assert len(outputs) == 20, f"Expected 20 outputs, got {len(outputs)}"
    print("  Bottleneck-based model test passed")
    
    # Test zero_init_residual option
    model_zero = AttributesResNet(BasicBlock, [2, 2, 2, 2], num_attributes=5, zero_init_residual=True)
    outputs_zero = model_zero(x)
    assert len(outputs_zero) == 5, f"Expected 5 outputs, got {len(outputs_zero)}"
    print("  Zero-init residual test passed")


def test_gradients():
    """Test gradient computation"""
    print("Testing gradient computation...")
    model = AttributesResNet18(num_attributes=2)
    x = ms.Tensor(mnp.ones((1, 3, 224, 224)), ms.float32)
    
    # Test that the model can compute gradients
    def forward_fn(x):
        outputs = model(x)
        # Sum all outputs for gradient computation
        loss = sum(out.sum() for out in outputs)
        return loss
    
    grad_fn = ms.grad(forward_fn)
    grads = grad_fn(x)
    assert grads.shape == x.shape, f"Expected gradient shape {x.shape}, got {grads.shape}"
    print("  Gradient computation test passed")


if __name__ == "__main__":
    print("="*50)
    print("Running CelebA ResNet MindSpore tests...")
    print("="*50)
    
    test_basic_block()
    test_bottleneck()
    test_attributes_resnet18()
    test_attributes_resnet_custom()
    test_gradients()
    
    print("="*50)
    print("All tests passed successfully!")
    print("="*50)