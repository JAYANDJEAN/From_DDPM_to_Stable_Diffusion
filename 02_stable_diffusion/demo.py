from stable_diffusion_pytorch import pipeline

prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts)
images[0].save('output.jpg')
