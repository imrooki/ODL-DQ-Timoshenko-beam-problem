#!/usr/bin/env python3
"""
Adaptive Learning Rate Scheduler (pure dynamic-threshold version)

Author: Yang
Version: 3.0

Functional description:
- Intelligent learning rate scheduling based on dynamic threshold calibration
- Automatically adapts to the loss magnitude of different boundary conditions
  (C-C, H-H, S-S, etc.)
- Three-stage management: early, mid, late
- Early/mid: dynamically adjust the learning rate based on the loss improvement rate
- Late: fixed learning rate to ensure stable convergence

Core algorithm:
    After collecting loss data during the warmup period (warmup_epochs), compute
    the dynamic thresholds:
    loss_range = initial_loss - min_warmup_loss
    early_threshold = initial_loss - (loss_range x early_ratio)
    mid_threshold = initial_loss - (loss_range x mid_ratio)

Design principles:
- Fully controlled by parameters, with no hard-coded thresholds
- Generality: whether the initial loss is +10 or -0.0001, the stage is judged correctly
- Supports independent scheduling for two models (linear/nonlinear)
- Lightweight and easy to integrate into existing training loops
"""

from typing import Optional
import torch


class LossBasedAdaptiveLRScheduler:
    """
    Adaptive learning rate scheduler based on dynamic threshold calibration

    Core features:
    1. Dynamic threshold calibration: automatically detect the loss range during the warmup period and compute the relative thresholds
    2. Three-stage management: early -> mid -> late (one-way progression)
    3. Early/mid: dynamically adjust lr based on the loss improvement rate
    4. Late: fixed lr, no longer changes
    5. LR clamping: automatically adjust lr to the reasonable range of the new stage upon stage transition

    Usage example:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = LossBasedAdaptiveLRScheduler(
            optimizer,
            lr_early_max=1e-3,
            lr_early_min=2e-4,
            lr_mid_max=2e-4,
            lr_mid_min=1e-4,
            lr_late_fixed=1e-4,
            warmup_epochs=100,
            early_ratio=0.2,
            mid_ratio=0.6
        )

        for epoch in range(epochs):
            loss = train_one_epoch()
            scheduler.step(loss)
            current_lr = scheduler.get_lr()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_early_max: float = 1e-3,
        lr_early_min: float = 2e-4,
        lr_mid_max: float = 2e-4,
        lr_mid_min: float = 1e-4,
        lr_late_fixed: float = 1e-4,
        patience: int = 500,
        improvement_threshold: float = 1e-6,
        lr_decay_factor: float = 0.5,
        verbose: bool = False,
        # Dynamic threshold calibration parameters
        warmup_epochs: int = 100,
        early_ratio: float = 0.6,
        mid_ratio: float = 0.85,
        # Minimum stage duration (prevents entering the late stage too early under fast convergence)
        min_early_epochs: int = 1000,
        min_mid_epochs: int = 2000
    ):
        """
        Initialize the learning rate scheduler

        Args:
            optimizer: PyTorch optimizer instance
            lr_early_max: Early-stage maximum learning rate
            lr_early_min: Early-stage minimum learning rate
            lr_mid_max: Mid-stage maximum learning rate
            lr_mid_min: Mid-stage minimum learning rate
            lr_late_fixed: Late-stage fixed learning rate
            patience: Loss improvement monitoring window (epochs)
            improvement_threshold: Threshold for judging an "improvement"
            lr_decay_factor: lr decay factor (between 0 and 1)
            verbose: Whether to print lr change information
            warmup_epochs: Number of warmup epochs, used to collect loss distribution information
            early_ratio: Relative improvement-rate threshold for early->mid (e.g. 0.6 means transition at 60% improvement)
            mid_ratio: Relative improvement-rate threshold for mid->late (e.g. 0.85 means transition at 85% improvement)
            min_early_epochs: Minimum number of epochs for the early stage (do not enter mid early even if loss is met)
            min_mid_epochs: Minimum number of epochs for the mid stage (cumulative, counted from the start of training)
        """
        self.optimizer = optimizer
        self.lr_early_max = lr_early_max
        self.lr_early_min = lr_early_min
        self.lr_mid_max = lr_mid_max
        self.lr_mid_min = lr_mid_min
        self.lr_late_fixed = lr_late_fixed
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.lr_decay_factor = lr_decay_factor
        self.verbose = verbose

        # Dynamic threshold calibration parameters
        self.warmup_epochs = warmup_epochs
        self.early_ratio = early_ratio
        self.mid_ratio = mid_ratio

        # Minimum stage duration
        self.min_early_epochs = min_early_epochs
        self.min_mid_epochs = min_mid_epochs

        # Internal state
        self.loss_history = []
        self.current_lr = lr_early_max  # Initial lr is the early-stage maximum
        self.current_phase = "early"    # early / mid / late
        self.step_count = 0

        # Dynamic threshold calibration state
        self.initial_loss = None              # Record the first loss
        self.warmup_losses = []               # Warmup-period loss record
        self.dynamic_threshold_early = None   # Dynamically computed early threshold
        self.dynamic_threshold_mid = None     # Dynamically computed mid threshold
        self.warmup_complete = False          # Whether the warmup period is complete

        # Set the initial lr
        self._set_lr(self.current_lr)

    def _calibrate_dynamic_thresholds(self) -> None:
        """
        After the warmup period ends, calibrate the dynamic thresholds based on the loss distribution

        Core formula:
            loss_range = initial_loss - min_warmup_loss
            early_threshold = initial_loss - (loss_range x early_ratio)
            mid_threshold = initial_loss - (loss_range x mid_ratio)

        This ensures that, whether the initial loss is +10 or -0.0001, the stage transition is triggered at the same "relative progress"
        """
        if len(self.warmup_losses) < self.warmup_epochs:
            return

        initial_loss = self.warmup_losses[0]
        min_warmup_loss = min(self.warmup_losses)
        loss_range = initial_loss - min_warmup_loss

        if abs(loss_range) > 1e-10:
            # Dynamically compute the thresholds: threshold = initial loss - (loss range x improvement rate)
            self.dynamic_threshold_early = initial_loss - loss_range * self.early_ratio
            self.dynamic_threshold_mid = initial_loss - loss_range * self.mid_ratio
        else:
            # When the loss does not change, use a default threshold based on the initial loss
            # Assume the total improvement margin is the absolute value of the initial loss
            default_range = abs(initial_loss) if abs(initial_loss) > 1e-10 else 0.01
            self.dynamic_threshold_early = initial_loss - default_range * self.early_ratio
            self.dynamic_threshold_mid = initial_loss - default_range * self.mid_ratio
            if self.verbose:
                print(f"[LR Scheduler] Warning: loss_range ≈ 0, using default range: {default_range:.4e}")

        self.warmup_complete = True

        if self.verbose:
            print(f"[LR Scheduler] Dynamic thresholds calibrated after {self.warmup_epochs} epochs:")
            print(f"   Initial loss: {initial_loss:.4e}")
            print(f"   Min warmup loss: {min_warmup_loss:.4e}")
            print(f"   Loss range: {loss_range:.4e}")
            print(f"   Early threshold ({self.early_ratio*100:.0f}%): {self.dynamic_threshold_early:.4e}")
            print(f"   Mid threshold ({self.mid_ratio*100:.0f}%): {self.dynamic_threshold_mid:.4e}")
            print(f"   Min stage epochs: early≥{self.min_early_epochs}, mid≥{self.min_mid_epochs} (cumulative)")

    def _determine_phase_forward_only(self, current_loss: float) -> str:
        """
        One-way stage determination - can only advance, never go back

        Stage order: early -> mid -> late
        Once the next stage is entered it is fixed and does not revert

        Constraints:
        1. Stay in the early stage while the warmup period is not complete
        2. Even if the loss is met, the minimum stage duration must be satisfied
        """
        # Warmup period not complete, stay in the early stage
        if not self.warmup_complete:
            return "early"

        threshold_early = self.dynamic_threshold_early
        threshold_mid = self.dynamic_threshold_mid

        # Stage determination logic (one-way progression + minimum duration constraint)
        if self.current_phase == "early":
            # The minimum early epochs must be satisfied before transitioning
            if self.step_count < self.min_early_epochs:
                return "early"
            # Currently in the early stage, check whether it should move to the mid or late stage
            if current_loss < threshold_mid and self.step_count >= self.min_mid_epochs:
                return "late"
            elif current_loss < threshold_early:
                return "mid"
            return "early"

        elif self.current_phase == "mid":
            # The minimum mid epochs (cumulative) must be satisfied before transitioning to late
            if self.step_count < self.min_mid_epochs:
                return "mid"
            # Currently in the mid stage, can only advance to the late stage, cannot revert to the early stage
            if current_loss < threshold_mid:
                return "late"
            return "mid"

        else:  # late
            # Currently in the late stage, fixed and unchanged
            return "late"

    def step(self, current_loss: float) -> float:
        """
        Update the learning rate based on the current loss

        Args:
            current_loss: Loss value of the current epoch

        Returns:
            The updated learning rate
        """
        self.step_count += 1

        # Record the initial loss
        if self.initial_loss is None:
            self.initial_loss = current_loss

        # Dynamic threshold calibration - collect loss during the warmup period
        if not self.warmup_complete:
            self.warmup_losses.append(current_loss)
            if len(self.warmup_losses) >= self.warmup_epochs:
                self._calibrate_dynamic_thresholds()

        # One-way stage determination (can only advance, never go back)
        phase = self._determine_phase_forward_only(current_loss)

        # Detect a stage transition
        if phase != self.current_phase:
            if self.verbose:
                print(f"[LR Scheduler] Phase transition: {self.current_phase} → {phase} at step {self.step_count}")
            self.current_phase = phase
            # Reset the history upon a stage transition
            self.loss_history = []

            # LR clamping upon a stage transition
            # Ensure that after the stage transition the LR does not exceed the maximum of the new stage
            if phase == "mid" and self.current_lr > self.lr_mid_max:
                if self.verbose:
                    print(f"[LR Scheduler] Clamping LR from {self.current_lr:.2e} to {self.lr_mid_max:.2e} (mid phase max)")
                self.current_lr = self.lr_mid_max
                self._set_lr(self.current_lr)
            elif phase == "late" and self.current_lr > self.lr_late_fixed:
                if self.verbose:
                    print(f"[LR Scheduler] Clamping LR from {self.current_lr:.2e} to {self.lr_late_fixed:.2e} (late phase fixed)")
                self.current_lr = self.lr_late_fixed
                self._set_lr(self.current_lr)

        # Select the lr strategy based on the current actual stage
        if self.current_phase == "late":
            # Late stage: fixed lr
            new_lr = self.lr_late_fixed
        else:
            # Early or mid stage: adjust based on the loss change rate
            new_lr = self._adaptive_adjustment(current_loss, self.current_phase)

        # Update the optimizer's lr
        if new_lr != self.current_lr:
            self._set_lr(new_lr)
            if self.verbose:
                print(f"[LR Scheduler] Step {self.step_count}: lr changed {self.current_lr:.2e} → {new_lr:.2e}")
            self.current_lr = new_lr

        return new_lr

    def _adaptive_adjustment(self, current_loss: float, phase: str) -> float:
        """
        Adaptive adjustment strategy for the early and mid stages

        Args:
            current_loss: Current loss
            phase: Current stage ('early' or 'mid')

        Returns:
            The adjusted learning rate
        """
        # Add to the history record
        self.loss_history.append(current_loss)

        # Keep the history length bounded
        if len(self.loss_history) > self.patience:
            self.loss_history.pop(0)

        # If the history record is insufficient, keep the current lr
        if len(self.loss_history) < self.patience:
            return self.current_lr

        # Compute the improvement rate (the more negative the better, so use oldest - current)
        oldest_loss = self.loss_history[0]
        improvement_per_epoch = (oldest_loss - current_loss) / self.patience

        # Get the lr range based on the stage
        if phase == "early":
            lr_max = self.lr_early_max
            lr_min = self.lr_early_min
        else:  # mid
            lr_max = self.lr_mid_max
            lr_min = self.lr_mid_min

        # Determine whether the lr needs to be adjusted
        if improvement_per_epoch < self.improvement_threshold:
            # Loss improvement is slow or stagnant: reduce lr
            new_lr = self.current_lr * self.lr_decay_factor
            new_lr = max(new_lr, lr_min)  # Not below the minimum value
        else:
            # Loss improvement is good: keep the current lr
            new_lr = self.current_lr

        return new_lr

    def _set_lr(self, lr: float):
        """Set the optimizer's learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get the current learning rate"""
        return self.current_lr

    def get_phase(self) -> str:
        """Get the current training stage"""
        return self.current_phase

    def state_dict(self) -> dict:
        """Save the scheduler state (for checkpointing)"""
        return {
            'loss_history': self.loss_history,
            'current_lr': self.current_lr,
            'current_phase': self.current_phase,
            'step_count': self.step_count,
            'initial_loss': self.initial_loss,
            'warmup_losses': self.warmup_losses,
            'dynamic_threshold_early': self.dynamic_threshold_early,
            'dynamic_threshold_mid': self.dynamic_threshold_mid,
            'warmup_complete': self.warmup_complete
        }

    def load_state_dict(self, state_dict: dict):
        """Load the scheduler state (for resuming training)"""
        self.loss_history = state_dict['loss_history']
        self.current_lr = state_dict['current_lr']
        self.current_phase = state_dict['current_phase']
        self.step_count = state_dict['step_count']
        self.initial_loss = state_dict.get('initial_loss', None)
        self.warmup_losses = state_dict.get('warmup_losses', [])
        self.dynamic_threshold_early = state_dict.get('dynamic_threshold_early', None)
        self.dynamic_threshold_mid = state_dict.get('dynamic_threshold_mid', None)
        self.warmup_complete = state_dict.get('warmup_complete', False)
        self._set_lr(self.current_lr)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    use_adaptive_lr: bool = True,
    **kwargs
) -> Optional[LossBasedAdaptiveLRScheduler]:
    """
    Convenience factory function: create a learning rate scheduler

    Args:
        optimizer: PyTorch optimizer
        use_adaptive_lr: Whether to enable adaptive lr (returns None if False)
        **kwargs: Parameters passed to LossBasedAdaptiveLRScheduler

    Returns:
        Scheduler instance or None

    Example:
        scheduler = create_scheduler(
            optimizer,
            use_adaptive_lr=True,
            lr_early_max=1e-3,
            warmup_epochs=100,
            early_ratio=0.2,
            mid_ratio=0.6
        )

        if scheduler:
            scheduler.step(current_loss)
    """
    if not use_adaptive_lr:
        return None

    return LossBasedAdaptiveLRScheduler(optimizer, **kwargs)


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Adaptive LR Scheduler v3.0 (Pure Dynamic Threshold)")
    print("=" * 60)

    # Create a dummy model and optimizer
    import torch.nn as nn
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create the scheduler - pure dynamic threshold
    scheduler = LossBasedAdaptiveLRScheduler(
        optimizer,
        lr_early_max=1e-3,
        lr_early_min=2e-4,
        lr_mid_max=2e-4,
        lr_mid_min=1e-4,
        lr_late_fixed=1e-4,
        patience=10,  # Small value for testing
        improvement_threshold=1e-6,
        lr_decay_factor=0.5,
        verbose=True,
        warmup_epochs=5,   # Small value for testing
        early_ratio=0.2,
        mid_ratio=0.6
    )

    print("\n" + "=" * 60)
    print("Test 1: C-C boundary (loss starts high)")
    print("=" * 60)

    # Simulate the C-C boundary training process (loss starts high)
    simulated_losses_cc = [
        # Warmup period (5 epochs)
        0.1, 0.05, 0.02, 0.01, 0.005,
        # Warmup period ends, start using dynamic thresholds
        0.0, -0.005, -0.008, -0.009, -0.0095,
        -0.010, -0.0105, -0.0108, -0.0110, -0.0112
    ]

    print("\nSimulated Training (C-C):")
    for epoch, loss in enumerate(simulated_losses_cc, 1):
        scheduler.step(loss)
        print(f"Epoch {epoch:2d}: loss={loss:+.5f}, lr={scheduler.get_lr():.2e}, phase={scheduler.get_phase()}")

    print("\n" + "=" * 60)
    print("Test 2: H-H boundary (loss starts low)")
    print("=" * 60)

    # Create a new scheduler to simulate the H-H boundary
    optimizer2 = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler2 = LossBasedAdaptiveLRScheduler(
        optimizer2,
        lr_early_max=1e-3,
        lr_early_min=2e-4,
        lr_mid_max=2e-4,
        lr_mid_min=1e-4,
        lr_late_fixed=1e-4,
        patience=10,
        improvement_threshold=1e-6,
        lr_decay_factor=0.5,
        verbose=True,
        warmup_epochs=5,
        early_ratio=0.2,
        mid_ratio=0.6
    )

    # Simulate the H-H boundary training process (loss starts low)
    simulated_losses_hh = [
        # Warmup period (5 epochs) - loss starts low
        -0.00016, -0.00069, -0.00157, -0.00697, -0.00841,
        # Warmup period ends, start using dynamic thresholds
        -0.00842, -0.00843, -0.00844, -0.00845, -0.00846,
        -0.00847, -0.00848, -0.00849, -0.00850, -0.00851
    ]

    print("\nSimulated Training (H-H):")
    for epoch, loss in enumerate(simulated_losses_hh, 1):
        scheduler2.step(loss)
        print(f"Epoch {epoch:2d}: loss={loss:+.5f}, lr={scheduler2.get_lr():.2e}, phase={scheduler2.get_phase()}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
