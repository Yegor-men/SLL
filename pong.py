import pygame
import random


class PongGame:
    def __init__(self, screen_size=(800, 600)):
        """
        Initialize the Pong game for manual tick updates.
        """
        pygame.init()
        self.screen_width, self.screen_height = screen_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        # Paddle setup
        paddle_width, paddle_height = 10, 100
        paddle_x = 20
        paddle_y = (self.screen_height - paddle_height) // 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, paddle_width, paddle_height)
        # Track paddle Y as float for subpixel
        self._paddle_y = float(paddle_y)
        # Convert 5 px/frame @60fps -> px per second -> px per ms
        speed_per_sec = 5 * 60
        self.paddle_speed_ms = speed_per_sec / 1000.0

        # Ball setup
        ball_radius = 10
        # Track as floats for subpixel motion
        self.ball_x = self.screen_width / 2.0
        self.ball_y = self.screen_height / 2.0
        self.ball = pygame.Rect(
            int(self.ball_x - ball_radius),
            int(self.ball_y - ball_radius),
            ball_radius * 2,
            ball_radius * 2,
        )
        # Ball speed: 3 px/frame @60fps
        speed_per_sec_ball = 3 * 60
        self.ball_speed_x_ms = (
            speed_per_sec_ball * (1 if pygame.time.get_ticks() % 2 == 0 else -1)
        ) / 1000.0
        self.ball_speed_y_ms = (
            speed_per_sec_ball * (1 if pygame.time.get_ticks() % 3 == 0 else -1)
        ) / 1000.0

        # Monitoring variables
        self.ball_distance_x = 0
        self.ball_position = "middle"  # 'above', 'middle', 'below'
        self.paddle_hit = False
        self.ball_missed = False

        # Cooldown in ms (e.g. 1000ms = 1s)
        self.paddle_cooldown_ms = 0

        # Simulation time
        self.sim_time_ms = 0.0

        # For drawing at 60fps simulation
        self._accum_time_ms = 0.0
        self._frame_interval_ms = 1000.0 / 60.0

    def tick(self, dt_ms=1.0, move_up=False):
        """
        Advance the simulation by dt_ms milliseconds.
        move_up/move_down: control inputs for the paddle.
        """
        # Advance simulation clock
        self.sim_time_ms += dt_ms

        # Poll events (keep window responsive)
        pygame.event.pump()

        # Paddle movement via input flags
        if move_up and self._paddle_y > 0:
            self._paddle_y -= self.paddle_speed_ms * dt_ms
        if not move_up and self._paddle_y < self.screen_height - self.paddle.height:
            self._paddle_y += self.paddle_speed_ms * dt_ms
        # Apply to rect
        self.paddle.y = int(self._paddle_y)

        # Update ball position (float)
        self.ball_x += self.ball_speed_x_ms * dt_ms
        self.ball_y += self.ball_speed_y_ms * dt_ms
        # Assign to rect (center-based)
        self.ball.centerx = int(self.ball_x)
        self.ball.centery = int(self.ball_y)

        # Reset flags
        self.paddle_hit = False
        self.ball_missed = False

        # Wall collisions
        if self.ball.top <= 0 or self.ball.bottom >= self.screen_height:
            self.ball_speed_y_ms *= -1
        if self.ball.left <= 0:
            self.ball_speed_x_ms *= -1
            self.paddle_cooldown_ms = 1000
            self.ball_missed = True
        if self.ball.right >= self.screen_width:
            self.ball_speed_x_ms *= -1

        # Paddle collision if cooldown expired
        if (
            self.paddle_cooldown_ms <= 0
            and self.ball.colliderect(self.paddle)
            and self.ball_speed_x_ms < 0
        ):
            self.ball_speed_x_ms *= -1
            # Reposition
            self.ball_x = self.paddle.right + self.ball.width / 2
            self.ball.centerx = int(self.ball_x)
            self.paddle_hit = True

        # Update monitoring metrics
        self.ball_distance_x = abs(self.ball.centerx - self.paddle.centerx)

        if self.ball.centery < self.paddle.centery:
            self.ball_position = "above"
        elif self.ball.centery > self.paddle.centery:
            self.ball_position = "below"
        else:
            self.ball_position = "middle"

        # Cooldown countdown
        if self.paddle_cooldown_ms > 0:
            self.paddle_cooldown_ms -= dt_ms

        # Draw at simulated 60fps
        self._accum_time_ms += dt_ms
        if self._accum_time_ms >= self._frame_interval_ms:
            self._accum_time_ms -= self._frame_interval_ms
            self._draw()

    def _draw(self):
        """
        Internal: redraws the current frame.
        """
        self.screen.fill(self.BLACK)
        pygame.draw.rect(self.screen, self.WHITE, self.paddle)
        pygame.draw.ellipse(self.screen, self.WHITE, self.ball)
        pygame.display.flip()

    def get_simulation_time(self):
        """
        Returns the total elapsed simulation time in milliseconds.
        """
        return self.sim_time_ms

    def quit(self):
        """
        Clean up and close the game window.
        """
        pygame.quit()



# ========================================================================================================================

import torch
import matplotlib.pyplot as plt


def scale_tensor(
    tensor: torch.Tensor,
    fps: float = 50,
    min_freq: float = 1,
    max_freq: float = 10,
) -> torch.Tensor:
    """
    Scales a tensor so that given an FPS, a minimal and maximal firing frequency,
    The output tensor is clamped so that when converted to a spike input it will be noise + the actual image
    """
    min_clamp = min_freq / fps
    max_clamp = max_freq / fps
    return tensor * (max_clamp - min_clamp) + min_clamp


def convert_to_spikes(
    tensor: torch.Tensor,
):
    from snntorch import spikegen

    return spikegen.rate_conv(data=tensor)


def render_image(tensor: torch.Tensor) -> None:
    assert tensor.size(0) == 3, "there aren't 3 color channels"

    # Move channels to last, CPU & NumPy
    img = tensor.permute(1, 2, 0).cpu().numpy()

    plt.figure("Live Image")  # gives/uses a window named "Live Image"
    plt.clf()  # clear the current figure
    plt.imshow(img)
    plt.axis("off")
    plt.show(block=False)  # non‐blocking
    plt.pause(0.001)  # let GUI update


# Example usage:
import time
import numpy as np
import torch

game = PongGame()
try:
    while True:
        # model sets move_up/move_down each tick:
        # time.sleep(0.02)
        game.tick(dt_ms=20, move_up=True)
        surface = pygame.display.get_surface()
        arr = pygame.surfarray.array3d(surface)
        arr = np.transpose(arr, (2,1,0))      # → (3,H,W)
        tensor = torch.from_numpy(arr).float() / 255.0
        tensor = scale_tensor(tensor)
        tensor = convert_to_spikes(tensor)
        render_image(tensor)



except KeyboardInterrupt:
    game.quit()