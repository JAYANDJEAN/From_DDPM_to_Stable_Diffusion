from stable_diffusion_pytorch import pipeline

prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, do_cfg=False)
images[0].save('output.jpg')
