import cv2

from homeostatic_vision_ant_env import HomeostaticVisionAntEnv


def test_env_design():
    env = HomeostaticVisionAntEnv(render_mode="rgb_array", image_size=(512, 512))
    obs, info = env.reset()

    print("\nControls:")
    print("- Press 'q' to quit the visualization.")
    print("- The window shows the Ant's First-Person Perspective (POV).")

    try:
        while True:
            # Sample a random action
            action = env.action_space.sample()

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Extract the POV image - RGB part only for visualization
            pov_image = obs["vision"][:, :, :3]

            # Convert RGB to BGR for OpenCV display
            pov_bgr = cv2.cvtColor(pov_image, cv2.COLOR_RGB2BGR)

            # Display the POV
            cv2.imshow("Ant POV Perspective", pov_bgr)

            # Also show the side-view environment camera in another window
            env_bgr = cv2.cvtColor(info["environment"], cv2.COLOR_RGB2BGR)
            cv2.imshow("Global Environment View", env_bgr)

            # Wait for 1ms and check if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if terminated or truncated:
                # if terminated:
                #     print("Episode terminated. Resetting environment...")
                # else:
                #     print("Episode truncated. Resetting environment...")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
        cv2.destroyAllWindows()
        print("Environment closed.")


if __name__ == "__main__":
    test_env_design()
