Link to the app: <a src="https://chatbot-mu-seven.vercel.app/">https://chatbot-mu-seven.vercel.app/</a>

I built a simple talking chatbot with my own voice! His name is Bernard (duh) and if you talk to him, he will respond to you!

Tech stacks: React, Tensorflow.js, pytorch, NLTK, AWS S3 (CDN for model) 
Deployed on Vercel

TODO:
1) Add context - context is king
2) Fit more questions 

Current bugs to be fixed (maybe, not important):
1) Mobile doesn't work really well (for iPhone, Safari works tho)
2) Add a mute button/volume widget so people can mute the bot or lower the volume if they want 
3) Add a text option, so user can type instead of talk 

What I learned and what to be explored in the future:
1) For small businesses where they aren't a lot of questions to be asked, is it feasible that we just overfit the model? For large businesses like Amazon, how is this done properly? Pretty sure they don't overfit their chatbot, right?
2) More NLP technique
