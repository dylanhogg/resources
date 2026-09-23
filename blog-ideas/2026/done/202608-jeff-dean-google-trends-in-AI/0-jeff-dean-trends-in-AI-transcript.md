Lecture summary: Trends in AI by Jeff Dean

https://www.youtube.com/watch?v=UTTeXZrpMR0

0:12
good afternoon. Um, it's not 4:30 yet, but since the room is
0:19
already full, I'll just start doing introduction to save some time. Okay. Um
0:24
it's a real pleasure to introduce Jeffin a chief scientist for Google uh research
0:31
and the Google deep mind and also he's a co-lead of Gemini project.
0:37
I would say before the now famous Jeff Dean at Google era, I actually met Jeff
0:44
in the 90s when Jeff was in digital western research and so I can confirm is
0:51
not overnight success story. Jeff has been quietly except exceptional for a
0:57
very long time. Um Jeff's bio you probably read in the
1:03
flyer um reads a little bit like a guided tour of modern computing and uh
1:11
from largecale information retrieval operating system distributed system compilers and then if that wasn't enough
1:18
the second career is that AI machine you know ML accelerator vision model
1:26
low-level software I'm going to not upgrade Chrome right now [laughter]
1:32
and um and also um
1:37
distillation neuronet network uh architecture search large language
1:43
multimodel uh model and application ranging from chip design to healthcare
1:49
translation and so on. So one of my favorite details of Jeff uh is that he
1:56
did his undergraduate thesis on neuronet networks and which means two things I think.
2:02
First he worked on neuronet network before uh it was cool and second he has the
2:10
rare distinction being able to say I like neuronet network before Google
2:17
made it popular. Okay, maybe Jeff did that to himself. But um if you're
2:25
wondering what takes um uh to to be good to have a good research uh taste,
2:33
apparently the answer is start early to be Jeff. So
2:38
uh along the way Jeff has received the ACM prize in computing ite
2:44
medal. He's a ACM fellow, a member of US National Academy of Engineering and uh
2:52
American Academy of Arts and Sciences. So, we're very fortunate to have him um
2:58
to give us the distinguished lecture today. Please join me to welcome Jeff. [applause]
3:07
All right. [applause] Thank you very much for that kind introduction and for hosting me. I'm I'm unexpectedly in town and Kai said, "Hey,
3:14
you want to come by and give a talk?" So, I'm delighted to be here. Um, and I'm going to talk to you about, you
3:19
know, what's happening in AI? I don't know if you've noticed, but it's a thing these days. Um, and we have many people
3:26
at Google working on this uh area. So, I'm presenting the work of not just my work, but many many people at Google uh
3:32
and elsewhere. So, a few observations which may not
3:38
come as a huge surprise to you. So machine learning and AI have really changed our expectations of what is
3:44
possible with computers. You know if you think back 1015 years ago you know computers couldn't see very well. They
3:50
couldn't really understand speech all that well. They definitely couldn't understand language very well. And now we have tools that enable computers to
3:57
interact with people in a much more natural style because they understand the modalities that people understand
4:05
and want to communicate with. Um and along the way increasing scale of larger
4:11
amounts of compute, larger amounts of data and more larger scale models have
4:17
delivered better results nearly continuously for the last sort of 13 or
4:22
14 years. This has been the trend and seems to be holding up even even today.
4:28
Um in addition to scale, algorithmic and sort of model architecture improvements
4:34
have also provided massive improvements as well. And these two things multiply together, right? If you can have much
4:40
larger scale and much better model architectures that are able to learn more, you know, per floating point
4:47
operation, those things together mean instead of, you know, 20x from scale,
4:54
you have 20x from scale and 50x from algorithms and you end up with a thousandx better thing than you had not
5:00
that long ago. Um, also the kinds of computations we want to run and the hardware in which we want to run them
5:07
are changing dramatically. So four observations which are maybe not surprising. Uh first neural nets and
5:15
gradient descent. These are seem to be key building blocks for how do you actually build truly intelligent uh
5:21
computer systems. Um and they're not new ideas. They're things that have been around for a long time. They were invented in the 70s and 80s. Um and
5:29
there was actually a lot of excitement about them in the late 80s and early 90s. I first ran across them in a you
5:36
know intro lecture where I was introduced to neural networks roughly like this. Um you know we have this kind
5:41
of nice very abstracted away uh version of how we think real life neurons behave
5:48
where you have a bunch of inputs you do some processing and you decide are you going to fire or not and what strength
5:54
are you going to fire um and you have connections of these things. Um, and
6:00
then back propagation is a way is a really nice algorithm that enables you to adjust the weights on the edges of
6:07
all these uh connections in order to make uh sort of um the the model behave
6:14
more in the way that you want it to behave where you give it a bunch of examples and back propagation enables
6:20
you to make the the model learn uh the behavior you want.
6:26
So I saw these as an undergrad in 1990 and I got super excited and I said,"Oh
6:31
wow, that seems like a really good abstraction. Um, it feels like it's a trainable thing and it can learn almost
6:38
anything." Um, and at that time we were not using very large uh neural networks
6:43
because we had really pathetically wimpy computers. Um, but I said, "Oh, maybe if
6:49
we use the parallel computer in the department, I could we could train much bigger neural networks. That would be
6:55
great. Maybe we just need to use 32 computers instead of one and then we could train really impressive neural
7:01
nets. That would be nice." So, I did it. I asked the professor who introduced the neural networks topic in one of the
7:07
classes I was taking if I could do a senior thesis with him, Vipin Kumar. Uh, and um, he said, "Sure." So I played
7:14
around with two different approaches for training neural networks in parallel. One is uh this is what you would now
7:21
call model parallel and data parallel training. Uh but I called them pattern partitioned and pipelined approach. Um
7:29
and uh these speed up curves are interesting but they taper off because I
7:35
did the stupid thing. I didn't make the model bigger and bigger as I increase the number of browsers. I just made it,
7:40
you know, much harder to parallelize uh with a fixed size model. Um but I sort
7:46
of filed that away as like oh that's an interesting abstraction and it seems like something I should pay attention to
7:53
but then I ignored it for you know the next 25 years.
7:58
Um, okay. So now I'm going to give you a whirlwind tour of a b b b b b b b b b b
8:04
b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b bunch of advances in the field of ML that have happened in the last 15 years that are really all uh
8:11
you know combined together to produce today's you know most sophisticated models. Um how did these today's models
8:18
come to be? Well, it was through advancements like the ones I'm going to describe. [snorts]
8:24
So, the origin of my getting back to working on neural nets uh at Google was
8:30
I bumped into Andrew Ing who's a Stanford faculty member and I in one of
8:35
our micro kitchens and I said, "Oh, what are you doing here?" He's like, "I'm not sure yet." Uh because I just started a
8:42
week ago. Um but my students at Stanford are doing uh interesting things with uh
8:48
training uh neural networks for speech and vision. I'm like, "Oh, that's cool.
8:54
We should train really big neural networks." I said, "Uh, and so that was the origin because we actually had quite
9:00
a lot of computers in those days." And I said, "Oh, why don't we try training
9:05
very very large neural networks?" Um, and so we built a software system that enabled us to express neural network and
9:13
connectivity of those neural networks and do both model and uh data parallel
9:19
training across uh many many computers. Um and that enabled us to train neural
9:25
networks that were 50 to 100 times larger than what we the largest ones we could find anyone talking about. Um
9:33
and uh we uh had a a way of doing this
9:38
where we had um actually many many asynchronous replicas of the model uh
9:44
cop getting updates of what the current parameters of the model were you know churning away on their local batch of of
9:51
examples computing a gradient and sending it to this distributed set of parameter servers in order to update the
9:57
parameters of the model. This is because it's asynchronous definitely not the right mathematical thing because in the
10:03
meantime the model parameters have moved from the other replicas giving the the parameter servers updates to the
10:08
parameter model to the parameters. Um but it seemed to work so we were sort of happy anyway despite it being sort of uh
10:16
theoretically wrong. Um, and one of the first things we we
10:22
trained was a very large model to do completely unsupervised learning on 10
10:28
million random YouTube frames. Um, and interestingly, we had a kind of a a
10:34
locally connected u neural network for computer vision um that would use an a
10:40
reconstruction loss. So we try to take a high level representation of the neurons at the highest level of the model and
10:46
then try to reconstruct the pixels that have been uh given as input and um
10:53
because that bottleneck layer uh forced the model to sort of learn higher level abstractions uh than the actual raw
11:00
pixels. Um, what we ended up finding was at the top level of this this model, some of the neurons had learned to be
11:07
sensitive to whether or not there was a face of a cat in in the image. Some of them learned to be sensitive to whether
11:14
there was a face. Some were sensitive to the, you know, is this the outline of a
11:19
person. Um, all without being told any of these images contained any of those
11:25
kinds of objects. So it's uh kind of neat to see that you could learn these high level abstractions and
11:32
representations from a purely unsupervised objective. Um what we could
11:37
then do is use the initialization of this unsupervised training in order to train with a supervised method um to uh
11:47
uh try to do computer vision uh image classification. And we used the
11:52
unfortunately we used the 22,000 category version of imageet which is a
11:57
more thinly traded version than the 1000 category one that most people are most familiar with. Um but we got a 70%
12:04
relative improvement in imageet 22k state-of-the-art by initializing with
12:10
our unsupervised uh model and then doing supervised learning on the um you know
12:16
the training set for that for that data set which is quite a large relative improvement. So we were happy with that
12:22
and we s saw that scale really mattered. Um we also started to do a bunch of work
12:30
in using neural networks for for language tasks. And one of the things we found was that even relatively simple
12:38
models where you have a vector-based representation of words or phrases um so
12:45
that you have a nice you know thousand or five 500 dimensional representation
12:50
of each each word or token um and have relatively shallow models. So
12:58
in fact we had just the embedding vector and then we would use the embedding vector of the word to try to predict
13:05
nearby words. That was one of the approaches we tried. Uh and what we
13:10
found was that if you apply this to lots of text uh and use this training
13:15
objective, you end up with nearby words uh in the highdimensional space tend to
13:20
be related ones. So like cat and puma and tiger are all low distance in the highdimensional space. And also that
13:27
directions are meaningful. Uh so you you you go in the same direction to change the gender of a word in a gendered
13:34
language uh as uh regardless of what word you start with. King and queen versus man and woman.
13:43
Uh a year or two later uh my three of my colleagues Ilia Sutzgver, Oral Vignyals
13:49
and Quaclay uh this uh used a neural encoder using an LSTM to uh essentially
13:58
be able to learn an encoder for one sequence and then a decoder for a
14:04
different sequence. Um, and when you initialize the state of the LSTM by
14:11
running the encoder, that ends up sort of capturing the things that have
14:16
occurred in that input sequence and enables the model to then use that initial state to then do a good job of
14:23
predicting the the target sequence. Uh so the the canonical example use case of
14:28
this is language translation where maybe you have a bunch of English French sentences uh pairs that mean the same
14:35
thing and the input sentence is English and so when you're training you try to
14:40
decode the French sentence and you get a supervised learning objective for every
14:46
token or for every word and then when you're trying to translate a new sentence you just use the encoder to
14:52
encode the rep the LSTM state and then decode from that state in order to
14:57
produce the translated uh French sentence.
15:03
So as we saw more and more success of neural networks and in particular scaling up neural networks for problems
15:11
in computer vision, speech recognition and language, we were starting to
15:18
both get excited but also a bit worried about how we would actually deploy these much better models. Um, so this is kind
15:25
of the origin of how we started to create our own tensor processing unit
15:30
chip uh program is I did this back of the envelope calculation and said okay
15:35
we have this really nice high quality speech recognition model. It has the error rate for speech recognition which
15:42
is a huge improvement. It's sort of equivalent to 20 years of speech research in like six months since we
15:48
started the project by training just a very large you know particularly
15:54
sophisticated model architecture. It's like a eight layer fully connected neural net uh just trained on a lot of
16:01
data. And that back of the envelope calculation said if we wanted to deploy
16:06
this to a scenario where we had a billion users talking to this model 3
16:11
minutes a day, we would need to double the number of computers Google had. And that didn't seem very tenable for
16:17
rolling out a a better speech recognition system. Um so that really convinced us that exploring
16:25
uh you know much more specialized computational devices for doing neural
16:30
network inference and later neural network training was going to be a a fruitful path. Um and so in you know
16:38
this sort of bootstrap got us to really approach that with conviction and
16:43
started up a chip design team in order to build these these chips. And then the first chip emerged two years later as
16:51
hardware projects tend to do. They take a little while but we eventually produced the TPUv1 chip um which had
17:00
took advantage of two really nice properties of of neural networks. The [clears throat] first is that
17:06
reduced precision is fine especially for inference. Um so uh you're perfectly
17:12
happy with you know about seven or eight bits of precision not 16 or 32 or 64 as
17:21
most people are thinking about in terms of comput floatingoint computation things. Uh and also that nearly all of
17:28
the models we were exploring were made up of different combinations of a
17:34
handful of specific kinds of uh linear algebra operations. you know things like
17:39
um matrix multiplies are at the heart of them and then certain kinds of dotproducts and normalization and
17:44
scaling. But if you think about most of the machine learning algorithms of today, different rearrangements of those
17:51
kinds of operations mean that you can build an accelerator that is capable of,
17:57
you know, very high performance reduced precision linear algebra and that can be applicable to a pretty wide range of
18:03
neural networks. And it's much more efficient because that's all it needs to do. It doesn't need to do anything else.
18:09
It doesn't need to run Microsoft Word or Chrome or anything else. Uh and so TPUv1
18:15
turned out to be 15 to 30x faster than contemporary CPUs and GPUs and 30 to 80
18:23
times more energy efficient uh than contemporary CPUs and GPUs. Um and it's
18:28
also now the most cited paper in ISK's 50-y year history. Uh
18:35
so we then uh turned the chip design team's attention to the training problem
18:40
because uh we felt like being able to scale up training more than we were able to on on CPUs and GPUs was going to be
18:48
important. And so this is where the uh focus has been for for quite a while.
18:53
And we've now produced many many generations of TPUs that are essentially
18:59
uh machine learning supercomputers, right? there uh it's more than just a single chip unlike TPUv1 instead it's
19:06
lots and lots of chips connected with a custom high-speed interconnect um that enables us to you know distribute model
19:13
training over those chips and communicate uh both uh activations and
19:19
parameter updates and gradient updates across that that custom interconnect.
19:24
Um, version four introduced this funky thing where we have an optical uh
19:30
interconnect for racks of these accelerators. So, there's a rack of 64 accelerators and we can make it seem as
19:37
though that rack of 64 accelerators is right next to another rack of 64
19:42
accelerators even though they're, you know, 100 meters across the data center floor. And if that rack fails, we can
19:48
swap in another rack and make it seem as though this rack is now right next to that one in terms of the net topology.
19:55
[snorts] Um, which is important. As you get more and more scale, you end up with failures.
20:01
And so you want to be able to reconfigure things like that. [snorts] And so we've seen, you know, continual
20:08
uh hardware performance scaling. So if you look at TPUv4 here on the left column, you know, the peak flops per
20:14
chip is about 275 teraflops. You know, the max pod size was uh 40 4,96 uh uh
20:24
chips and now we're sort of at roughly double that pod size and a much much more HPM memory and much more memory
20:30
bandwidth and many more flops per per chip. Uh if you compare the performance
20:35
peak performance per pod which is sort of the largest network configuration we
20:40
have of the the custom interconnect the TPUv2 pod is 1x and the most recent uh
20:48
sort of v6 but now we've stopped numbering them sadly it's harder to keep track of but Ironwood is the sixth
20:55
generation of FTPUs is about 3600 times the performance per pod FTP v2 and
21:02
it's also much more energy efficient So about 30x better uh energy uh you know
21:07
flops per watt which is what you ultimately care about in terms of u sort
21:13
of energy cost.
21:19
The other thing that has happened in the last 15 years is open source tools have really enabled the whole community to
21:25
build using the same pieces of abstraction and and software libraries and the community can work on sort of
21:32
improving those tools as well. So we developed a a series uh a software called TensorFlow based on our initial
21:39
lessons from the sort of 2011 to 2014 era with an internal uh software library
21:47
that we were using to to parallelize computations across um you know many computers and train large models and and
21:54
do inference. Um and so and then more recently uh various people at Meta and
22:00
other elsewhere uh introduced PyTorch and then uh another group of people at
22:05
Google have introduced a more functional uh style paradigm uh computation called
22:11
Jax. [snorts] Um all of these are are great systems and you know we have been using Jax for
22:17
our Gemini training uh and it's been great. Um and then in 2017 a number of my
22:25
colleagues uh uh worked on uh coming up with a better architecture than the sort
22:32
of recurrent LSTMs that were kind of the the state-of-the-art models for language
22:39
uh at the time. And the observation they had was really in an LSTM you you have a
22:46
vector of state uh uh of sort of the current representation of the state and
22:52
then when you advance one word at a one token at a time you update that state and there's a sophisticated uh thing
22:59
that uh sort of updates the state and then you go on to the next word and you update the state again and so as you
23:07
progress through say a hundred words of a sentence you update the state a 100 times. Um, and there's two problems with
23:13
that. One is you have this sequential process. Every word you have to wait for
23:19
the up update of the state and there's a sequential dependency on that state being ready before you can go on to
23:26
update the state for the next word. So, it's not very parallelizable. Uh, which is not good on modern hardware. And the
23:32
other thing is you're trying to force all the things you went through into a
23:37
single vector that you have all the all the things you might want to remember from the things you went through uh in a
23:44
single vector rather than just saving all the vectors you went through and then being able to look at them uh as as
23:51
you wish uh later in the sort of language modeling process.
23:56
So the key observation in the transformer model is let's save all those vectors or some representation of
24:03
those vectors and then pay attention to them in a way where we have a learnable attention mechanism.
24:08
[snorts] And what they were able to to show was that compared to LSTMs of one, two or
24:15
four layers in red there, uh the test loss on the y-axis um went down
24:22
considerably using a transformer. and and no matter what model scale you looked at. So sort of a 10 to 100x less
24:30
compute and 10 times smaller model gave you uh with a transformer would give you
24:35
roughly the same quality as as you would get with an LCM that's much larger and much more computensive.
24:45
In 2018, you know, I think people really started to take seriously the idea, well, I guess in the word devec work, we
24:52
were already sort of doing that. But in 2018, really looking at how could you do
24:57
language modeling at scale with self-supervised data and just the observation there's a lot of text in the
25:03
world. And so self-supervised learning on this text can give you almost unlimited amounts of training data where
25:10
you know the right answer, right? like you have text and then you hide the next word and you try to guess the next word
25:16
and if you guess it correctly that gives you get a sort of a one and if you guess
25:22
it wrong that gives you a clear error signal that you can propagate into improving the model so that uh in such
25:28
settings you will guess better next time and so this is really one of the major
25:33
reasons that these chat and language models have gotten uh yeah I created a custom example for this week [laughter]
25:42
Um, and so there are two kinds of training objectives that people typically use.
25:47
One is called auto reggressive where you get to look at the prefix everything to the left uh and try to predict the next
25:54
word. So Princeton blank, you know, that that word is a little bit hard to predict. Could be Princeton New Jersey,
26:00
could be Princeton University. Um, you know, Princeton University blank. That one's hard to predict as well. Uh, is
26:08
cold in the blank. that that BV is a little easier to predict. Um so, uh
26:14
that's auto reggressive and then fill in the blank uh is where you get to have a
26:20
training objective where you just hide you get to look in both directions and you hide some of the words and you try
26:27
to force the model to guess the missing words uh that you've hidden. And again,
26:33
you get this nice uh sophisticated um you know uh very very crisp uh error
26:40
signal that you can use to train the model. And you can just train on lots and lots of text either e either of the objectives. Um the fill-in- thelank one
26:47
tends to be used to build representations that some that give you a good representation of a bunch of
26:52
text, but you can't really use it for a chatbot because you don't have the things to the right. So uh you have to
26:58
sort of end up using mostly auto reggressive models for conversational applications but fill and blank also has
27:04
some good uses. Um in 2021 a number of other of my
27:09
colleagues uh applied transformers to computer vision. At that time the sort of state-of-the-art was to use various
27:16
kinds of convolutional models which have these kind of local receptive fields and then you build up and share the
27:22
parameters among different positions in that uh in the U image. Um and instead
27:29
you can just apply a transformer uh based model to the same same problem.
27:35
And what they found was again you get a much better you know if you look at the
27:40
TPU core days to reach certain levels of accuracy um the TPU core days are much
27:46
lower and the accuracy is higher uh compared with say ResNet 152 which was
27:52
kind of a good state-of-the-art model at that time.
27:59
And the other nice thing is you can kind of see what the attention mechanism is paying attention to. So it gives you a
28:04
little bit of model interpretation of like what is it paying attention to in order to make a say a classification
28:09
prediction that that middle one is an airplane. [clears throat]
28:14
Another area that we have done a bunch of work in including myself is in sparse models. Uh you know in a dense model you
28:23
have a bunch of parameters in the model and every inference or every token you activate the entire model in order to
28:29
make the prediction for for that example. Um, and that that seems kind of
28:36
uh that's not how our brains work, right? We have very different pieces of our brains that are useful for different
28:41
kinds of things. And we activate the right part. So while I'm, you know,
28:46
worried about the garbage truck backing up at my car, some parts of my brain are active and the part that thinks about
28:52
Shakespearean poetry is not active. So spar with sparse models the idea is
28:59
we want to have different pieces of the model that are good at different kinds of things. Um and then be able to
29:05
automatically learn not just those experts but also learn the routing mechanism to learn which pieces of the
29:12
model to actually activate for those different examples. And so it gives the model much larger capacity but still
29:20
keeps the inference cost relatively low because you're only activating, you know, a small number of the parameters in the model. A and importantly, it
29:27
gives you an 8x reduction in training compute for the same accuracy. That's uh
29:33
choice A there. Or you could choose to spend the same amount on training the
29:39
model [snorts] uh and [clears throat] get a much better model, right? Same amount of compute. That that would be choice B there. And
29:45
often these kinds of things you you choose a bit of A and B and end up kind of somewhere there. Um but this has been
29:51
a really important tool for making our model better and better over time is having sparity be part of the mix. Uh
29:58
there's a whole litany of work that uh both I and so and my colleagues have
30:04
done in this general area looking at various aspects of how to do the routing you know how many experts do you
30:10
activate um and so on.
30:16
[clears throat] Another thing we've been working on is building up uh abstractions for
30:22
distributed ML computations. So, Pathways is a system we started working on quite a while ago that really wants
30:29
to abstract away this pile of accelerators underneath your computation
30:34
and give you the sort of researcher the illusion that you just have a giant
30:40
computer with tens of thousands of chips attached to it and you can mostly just focus on the machine learning
30:46
computations you're trying to express. And underneath the covers, Pathways does a bunch of work to manage how to
30:54
communicate uh the machine learning computation you've expressed uh amongst
30:59
all these chips. And so it will for example use within one of these pods, it
31:05
will use the custom TPU interconnect. When you go across the pod boundary in the same data center building, it will
31:11
use the data center network. uh when you go across buildings in the same data center campus, it will use the purple
31:18
network link. And when you go go across, you know, perhaps widely distributed metro regions, it will use the um sort
31:27
of wide wide area network uh links. And [clears throat] then when you
31:32
actually put jacks on top of pathways, what that means is that we can drive the
31:37
entire training process from a single Python process on one host. So that that Python process says, "Oh great, I have
31:44
access to you know these these many thousands of TPU chips and um I can run
31:49
this computation and pathways kind of deals with what happens when different pieces of the system fail uh and and
31:57
swapping in new parts uh new new hardware under the covers.
32:03
Uh, another thing that tends to happen when you run at increasingly large scale
32:09
is sadly not all chips or machines or network cards or cables work as
32:15
designed. That's a very sad but true state of the world. Um, and in
32:21
particular, some of them don't just fail in ways that you can just like detect them immediately and and replace them.
32:28
they sometimes non-deterministically sort of produce incorrect results um silently and sometimes that's related to
32:36
dynamic conditions like the temperature of the rack at that particular moment.
32:41
Um, this is really challenging when you're running independent compute on these things, but it's really worse when
32:47
you're trying to have all these chips perform as a single orchestrated uh, you
32:54
know, machine learning uh, training computation where everything is is synchronous and this can quickly spread
32:59
bad results. So like if one of these things flips one of the bits in the exponent field of your gradient all of a
33:07
sudden you'll get you know 10 to the 20th propagating as a gradient instead of 02 and that makes you very very sad.
33:15
[clears throat] Um and so we monitor during the training
33:20
process all kinds of metrics about the health of our model. And in particular, you can look at, for example, the norm
33:26
of the gradient at every layer of the model and see are you seeing spikes in
33:31
that. So here's an example of a spike uh due to one of these silent data corruption things. Um now not all spikes
33:39
are necessarily due to these kinds of machine issues. Here's another spike.
33:44
There is no SDC. This is just some particular batch of examples that contain something that caused a very
33:50
large gradient uh to occur. Um, and it's a little hard to discern those things.
33:57
Um, so one of the ways we do this is we have deterministic replay that we
34:02
trigger automatically. So if we see a gradient spike, then we will effectively
34:08
replay that step or the last few steps and see if we get the same answer. And if we get the same answer, we conclude,
34:15
well, that's probably okay. That was probably due to the data, not due to
34:20
hardware errors. But if the hardware errors are uh you know causing the issue
34:25
then hopefully we get a different answer the next time we replay it.
34:33
And you can also have cases where we don't get a gradient spike
34:38
uh but there is actually silent data corruption happening um but it doesn't
34:44
cause a very large spike in the gradient and so that is sort of benign to the model even though we didn't detect it.
34:51
So all kinds of things can happen. It's very fun.
34:56
Uh the pathway system is part of the way in which we transparently handle these
35:02
things. So for example, let's say these blue uh pods or slices of pods are
35:10
currently part of our synchronous training jobs. We have these seven blue things. Um, we have a hot spare in gray
35:17
and then we have another uh hot spare and we're running a SDC checker. So, we
35:24
have a, you know, a continuous thing that we're running even on the idle idle
35:29
chips in order to try to weed out bad hardware uh before we're actually starting to use it. Um, the defective
35:37
machine causes one of these silent data corruptions. The uh SDC checker
35:42
identifies this because we replayed and detected that um there was a mismatch in
35:47
the answers when we ran them twice. Uh and we can pinpoint which which uh which
35:52
pod it is and which which chips. We can evict that thing and then use the hot
35:58
spare uh to now replace it transparently in the training process and continue on
36:03
our merry way. send that uh that thing to our repairs team and hopefully they
36:10
can figure out in more detail what's going on with it. [snorts]
36:16
Okay. In 2022, uh thinking longer at inference time
36:21
became more of a thing. uh and in particular the observation was that if you prompt the model to show more of its
36:29
work um on an example problem and then ask it to solve a similar kind of
36:34
problem. You know, if you just asked the answer, please give me the an model just
36:39
give me the answer. You know, in this example, you gave you prompted the model with one example problem and you said
36:45
the answer is nine. And then you gave it the problem you really care about and the model will sort of model that
36:51
behavior and say that the answer is 50 and it actually gets this problem wrong. But if you give the model a worked out
36:58
example, then the model uh will tend to work out in the same similar way the
37:06
individual steps to get through the actual problem solving process and give
37:11
you be more likely to give you the correct answer, right? Which is kind of nice. This is what your fourth grade math teacher told you and it's nice to
37:18
see that ML models behave the same way. Um and uh one way to view this is you're
37:26
actually giving the model more compute to think at inference time, right? Because every token it needs to generate
37:32
is a pass through the model. And so you can think of the length of the answer as
37:37
some proportional aspect of how much uh computed devoted to to producing the
37:42
answer. Uh and what you see is that as the model gets to a certain scale, when
37:47
you allow it to do chain of thought, you you encourage it with chain of thought
37:52
prompting, you know, you get pretty interest large increases in the solving
37:58
rate of GSMK. Remember, this is like eighth grade math problems like the ones here. John, you know, John has five
38:05
rabbits. He gets two more. Uh maybe that's not eighth grade, that should more like fifth grade or second grade.
38:11
But anyway, um, you know, these are pretty simple problems, but we were very excited in 2022 about high rates of of
38:18
accuracy improvements on those kinds of problems. Remember that. [snorts]
38:24
Another thing we worked on uh was distillation where you can effectively
38:29
use a powerful teacher model to make a smaller and cheaper student model from
38:36
the uh you know the more sophisticated predictions that the larger model can
38:41
make. Um and in particular like if you're doing language modeling perform the conerto for blank right so the true
38:48
answer in this case was violin. um and that's what your your training methodology would would use as the
38:55
signal to give you a supervised learning signal. But if you have a really good
39:00
model already, what that teacher model can give you is a probability distribution over the word the likely
39:09
next words. So it could be violin, could be piano, could be trumpet. It's almost certainly not airplane, right? Unless
39:16
there's new concertos I don't know about. Um, and so that gives that gives
39:21
a really rich gradient signal at the very top of the model that gives it much more information than just did you get
39:27
violin or not. Uh, which is what the the sort of nondistilled uh, loss would give
39:33
you. And so this enables you to get a lot of the capability of a very large
39:38
model into a much smaller model and also to do so with much less data perhaps.
39:43
So in the initial experiments in the paper, we were looking at a speech uh recognition problem. And so we had a
39:49
baseline where we used 100% of the training set and got a training frame accuracy of 63.4% and a test frame
39:57
accuracy of 58.9%. But [snorts] if you drop the baseline to use just 3% of the training data, then
40:05
you overfit on that. So you have 67% training frame accuracy but your test
40:10
frame accuracy plummets to 44.5%. Um but if you use the soft targets with
40:16
distillation you get uh you know quite a nice uh training frame accuracy actually even
40:23
better than the baseline on the 100% of the training data with just 3% of the training data and your test frame
40:28
accuracy is nearly as good as the full model trained on 100% 100% of the data.
40:34
So this is a really key technique for effectively enabling you to uh morph any
40:40
shaped model into another shape model often a larger model into a smaller model. This is for example what we use
40:47
in Gemini to go from the pros scale model into a flashcale model that is almost as good as the proscale model.
40:57
And then in the last you know four or five years I would say uh reinforcement
41:02
learning has become much more important in eliciting the desired properties that these models have. Um and so given a
41:09
pre-trained model where you've just uh exposed it to a whole bunch of training data of text um that model has a bunch
41:17
of uh capabilities in it but they often you want to help steer the capabilities
41:24
of these models a bit. So in particular, you want to maybe encourage behaviors that you want in the way in which the
41:30
model responds. You might like a certain style of it uh of response. You might like to prefer certain length responses
41:38
for certain kinds of things. You might want safety properties to be uh put into the model. So don't engage in these
41:44
kinds of topics or or say something like this in these set scenarios. And you
41:49
also want to enhance the capabilities by showing the model how to tackle more complex problems. And so one of the key
41:56
cr questions when you use uh reinforcement learning is where does the reward signal come from? Um but these do
42:03
dramatically improve the model capability in a lot of domains. And the reward signals turns out can come from
42:09
lots of different places. So one is you can use human feedback. you know, do you
42:14
like A or B as an answer for this kind of question and have humans evaluate
42:20
that and give a reward and then have the model um or is this a good answer for
42:27
this question? Uh they can say yes or no. Those kinds of things can be very effective at making the model's behavior
42:35
u change pretty dramatically with even relatively few examples. You can also use uh reinforcement learning from
42:41
machine feedback where you have machine feedback from a different model. So often called a reward model maybe you
42:48
can prompt the reward model saying please say whether you prefer response A or B for question Q and then you
42:55
generate two responses from your base model and then you ask the reward model which one do you like better and that
43:00
gives you a reward signal. Um you can also do RL and verifiable domains like
43:06
math or coding and you can for example generate proofs or solutions to be
43:11
checked with the theorem prover that give and give a positive reward when the model you know uh produces a correct
43:17
proof for the problem you're trying to solve. You can generate code for coding things where you now have a reward for
43:24
does the code even compile? Um does the code compile and pass the unit test? You get more reward. Um these kinds of
43:31
things. So these really help a lot in both math and math and coding but also in kind of adjacent areas like more
43:37
sophisticated reasoning and planning in um you know adjacent domains. Um I think
43:43
there's a nice open research question of how do you improve the effectiveness of RL in non-verifiable domains and where
43:49
do you get interesting reward signals from? Uh those are those are things that I think are pretty pretty open and ripe
43:56
for lots of cool approaches. [clears throat] In 2023,
44:02
a few of my colleagues came up with this really nice method for doing spec making inference faster. Um one of the problems
44:09
with the auto reggressive decoding is in order to have the attention mechanism
44:16
have the state it needs when you're when you're prefilling. So if you have an input of a thousand tokens that you can
44:22
do completely in parallel. But now if you're going to try to start generating a response, you generate the first token
44:29
and then when you finish generating the first token, you can now look at that token and be able to now generate the
44:35
next token because you need the attention state for that previous token in order to actually generate the the
44:42
next token. So it's a very again a sequential process a little bit like the problem we had with LSTM's where you
44:47
have this sequential decoding phase. Um and so speculative decoding is a nice uh
44:54
approach and you change only the decoding algorithm. You don't have to retrain your model. You have no architecture changes, no retraining and
45:01
you're guaranteed to have the identical output distribution. Okay. So the observation is decoding from very large
45:08
transformer models is memory bound. So the hardware can do way more floatingoint operations than it can do
45:15
transfer all the weights of your model into the MX into the multiplier unit in order to do the single token of decoding
45:23
you want to be able to do. Uh observation two is that some tokens are a lot easier to predict than others. Um
45:30
so can you tell me what the square root of seven? Sure the square root of seven is blank. Right? That last token is
45:37
really really hard for the model to predict. But a lot of the other ones are very easy for the model to predict. Like
45:43
after the question mark, it can probably make a prediction of four or five tokens
45:48
and get uh some of them uh correct. In particular, some prefix of them correct.
45:54
And so the idea is you have a fast drafter model where you don't need the full sophistication of a much larger
45:59
model and you can quickly generate the next say eight tokens and then have the
46:05
target large model check them in parallel. This effectively gives you a batch size now of instead of one a batch
46:13
size of eight. And if you accept on average four and a half of those eight
46:19
tokens, then you've made your uh sort of um compute efficiency much much better.
46:26
Um so here's an animation of how this works. If you just try decoding with a large slow model at the top, you're
46:31
decoding one token at a time. And instead if you have a small model at the
46:36
bottom that's drafting you know sequences of four tokens at a time uh
46:42
and then the larger model I'm going to run that again. Okay. So now watch the bottom thing. The drafter is gen
46:49
iterating the yellow things and then the uh
46:56
the upper model is accepting the one the prefix of them that is correct according
47:01
to what it wanted to generate based on its own probability distribution.
47:07
So this actually helps quite a lot in in you know improving the efficiency of
47:12
inference in these models. Okay. So a whole bunch of things have come together in order to actually do uh make these uh
47:20
modern models work well. Uh you know lots of innovations in the inference algorithms in the training algorithms,
47:26
model architecture, software abstractions and in the hardware.
47:33
Okay. So now we get to Gemini. So we actually the the sort of origin of
47:41
Gemini was he actually had multiple separate groups at Google looking at
47:46
language models looking at sort of creating multimmodal models of various kinds and I I sort of said this is
47:54
stupid why don't we all work together and have instead of fragmenting our compute and our ideas let's all work
48:00
together and build a single kind of model that is multimmodal that uh has
48:07
you know a lot comput investment behind a more singular effort and that's the
48:13
sort of origin of the Gemini project and so we started this in February 2023 with a bunch of collaborators from Google
48:19
DeepMind Google Research and the rest of Google um and our goal is really train the world's best multimodal models and
48:26
use them in all the places we can uh and so we've produced uh in that time a
48:32
number of different sort of iterations of Gemini generations one 1.5 two, 2.5,
48:38
and most recently three. Um, and one of the things we wanted was for
48:45
the model to be multimodal from the start. Uh, in particular, we think it's really important that it deal with
48:51
language, with audio, with images, and with video. uh as well as other nonhuman
48:56
modalities like um in more recent Gemini versions we've been including some small
49:02
amounts of say LAR data or robotic control data because you want to at
49:07
least expose the model to the idea that these kinds of data exist
49:12
and then uh initially in the early Gemini versions we had an image decoder and a text decoder uh more recently
49:20
we've added audio decoders and video video decoders so that we have a single unified model
49:27
that can reason over input sequences that are mixes of any of these
49:32
modalities. So you can give it like an image and some text that describes the image or what you want the model to pay
49:37
attention to and maybe you have video and some audio and you can also produce any of those things as output um in an
49:43
interleved way if you want. Um and as I said we're basically
49:50
building on lots of these different advances. So we use TPUs, we use model parallel and data parallel training and
49:56
cross data center training. We have fast and automated detection of the STC errors, we use pathways, we use jacks,
50:02
we use distributed representation of words similar to wordtovec or the
50:07
sequence of sequence work. We use transformers. We our models are sparse typically. We distill uh from larger
50:15
models into smaller models. We use chain of thought decoding and more sophisticated versions of that. uh
50:20
speculative decoding is really important. We use SFT and various kinds of RL uh and a bunch of other uh things
50:28
I didn't have time to talk about. Um one of the things we've been uh
50:33
working on for a while is making our context length uh large because we think the information in the context window
50:40
for the model is really useful. You know it unlike m uh tokens in the training
50:46
data where you've sort of taken trillions of tokens of training data and stirred them together into hundreds of
50:51
billions of parameters but it's all a little bit muddled and fuzzy. The information in the context window of the
50:57
model the input that you give it is very crisp because you haven't mixed it with anything else. is sort of like I I see
51:04
these 900 pages of text you gave me and I can like look at different parts of it
51:09
and summarize and combine and pull out little pieces of it when I'm decoding.
51:15
And so uh extending the context window to very large lengths has been something we've been uh working on and we think is
51:22
important. Um, you know, one of the things we've tried to keep up with in every uh release of
51:30
Gemini since the earliest ones is that we want our we have different scales of
51:35
models. We have a pros scale model which is the most the highest quality model. Um, and then we have a flash scale model
51:42
that is ideally uh as close to that proscale model as we can get it. Uh and
51:47
in particular, we also want that flash scale model to be better than the previous generations pro model, right?
51:54
Uh and so we've been able to keep that up for three, four generations now, which we're pretty happy with. Uh
52:00
because it means that the highest end thing that is perhaps slower and more
52:05
expensive for a lot of use cases suddenly becomes more affordable six months from now uh because it now has
52:12
that capability in the flash scale model. um you know Gemini 2.5 Pro uh we
52:19
released in March I guess uh with um quite good results
52:25
um we've also been working on you know mathematical problem solving and in in
52:32
July of last year we competed in the international mathematical olympi the IMO um and we got five out of six
52:40
problems correct which is a gold medal score we were quite quite happy with that Um and we did it uh we competed the
52:48
previous year with a more sophisticated kind of juryrigged mathematically
52:53
specialized model that used lean and theorem provers and various things like that. Um for this year we used just an
53:01
out-of-the-box Gemini pro scale model with a high inference thinking time
53:06
budget in order to tackle these problems. So we were pretty happy that you know we didn't have the sort of uh
53:14
and we also had in the previous year we had a different model for geometry problems and things like that. Um so we
53:21
were happy to have a single general purpose model that is the same one we offered to users um was able to solve uh
53:29
and get a gold medal at the IMO. just to give you a sense of, you know, how far we've come in mathematics. Remember back
53:36
to Fred had five rabbits and then he got two more. Okay, so this is one of the IMO
53:42
problems. Um, and you can see all the five correct
53:47
solutions of the problems we did there. Uh, but this is one of the problems. And so this is the output of our model.
53:59
And you can see this is sort of fairly sophisticated mathematical style writing and and formatting of proofs and so on.
54:08
And we uh established an upper bound. We had
54:14
another lema. We constructed and made a lower bound. And then we we proved that
54:20
f of n is less than or equal to 4n for all bonso functions which was part of the introduction of the problem. So like
54:28
this is now quite a bit beyond GSM8K I would say.
54:33
[snorts] I mean it's not research level mathematics yet but you can see the slope is quite quite large in two years.
54:42
Um and then our Gemini 3 model release uh happened in November uh of this year
54:48
where we were pretty happy with a lot of the benchmark results. Um but also we
54:54
have some uh you know we were happy with where it landed in you know one of the
54:59
ways that you evaluate these models in the wild is you can use this uh setup
55:05
called Ella Marina where people see [snorts] responses from two anonymized models and
55:11
then they say which one is better and from lots of pairwise for their
55:16
particular prompt right like people can go say can you please help me solve this uh mathematical problem or can you help
55:23
me produce a recipe for my you know French onion soup or something and uh
55:29
then people grade A or A or B which is better like going to the opthalmologist
55:35
uh and through a lot of those evaluations you can get a sense of the relative strength of different models uh
55:41
and that's where you get an ELO score a little bit like a a chess ranking
55:47
um so one of the things we've been working on is uh search and AI mode. We
55:53
actually want these models to be able to produce the UI that the user will see. Uh, I'm not sure this is going to work.
56:00
Am I on the Wi-Fi? I might not be on the Wi-Fi. This is going to make me sad.
56:07
All right, let's enable the hotspot. That's probably easier than logging into
56:12
the Princeton guest network, isn't it?
56:18
One moment. Sorry. Okay,
56:24
now we're gonna wait for that. Come on. Oh, there we go. Okay, great.
56:32
Awesome. Now, we're going to try it.
56:39
Go there. Go there. Yay. Thank you, T-Mobile. Okay. [laughter]
56:48
All right. So this is basically they put in a sort of researchy textbooky thing
56:54
about uh some biological thing and now they said show me how this works and
57:01
what it's done is created an interactive visualization of the material there and
57:07
you can use sliders to control the visualization and get a sense of the
57:12
topics you're trying to learn about which is pretty nice. You can see how this would be really useful for
57:18
educational things. Um, here's another example. What this couple
57:25
has is a bunch of recipes in English and Korean in their in handwritten pieces of
57:32
their Oh, I will actually since we have closed
57:38
captions, I will speed this up to 0.25.
57:44
Uh, and so they basically are taking things like that. They're asking Gemini to take all those recipes and say
57:52
translate and transcribe them in English and Korean. And then it has any Korean speakers can
58:00
maybe validate um, English translate. Okay, so now they're going to say, "Please create me
58:05
a bilingual website using these recipes." Okay, so there we go. And then we added
58:11
some automatically generated images and this is a website that was created from that. So I I mean I think one of
58:20
the things about this is that this kind of sophisticated model will enable a lot
58:26
of people who maybe don't have the skills to write software to get things
58:31
that they want in software form. So I think we're going to have a lot more software in the world than we have now.
58:37
Um, just kind of a cute example, but but I think it's good to pay attention to what these models can do and and the
58:44
sophistication level of the coding and and uh problem solving they can do. Um,
58:51
I mean, I'll skip that. Uh, generative models for images and for
58:56
video have also been uh um improving quite rapidly I would say. Um
59:04
so we had a initial release of something called nanoanana uh and then we made a
59:10
much more computationally expensive but higher quality version called nanobanana pro u
59:18
and so as examples of things you know you can start with that blueprint and say please make a realistic 3D image
59:24
from the blueprint. You can see how this would be quite useful for visualizing different architectural things. um you
59:33
can say please annotate the original transformer architectural diagram with the important things that happen in
59:38
different parts of the the diagram. You can see if you're reading a research paper um getting a visualization of that
59:46
uh could be pretty useful. Um you can actually view thinking in
59:52
pixel space. So here the input image is um you know this sort of ball and ramp
1:00:01
problem and tell me which bucket the ball lands in use images to solve it step by step right so you can now
1:00:08
actually see the model will generate a series of images of its reasoning. So it goes from there.
1:00:14
Step one, ball rolls there. Step two, the ball rolls on to three. And then uh
1:00:20
step three, the ball rolls on to five. And then step four, it ends up in bucket B. And so you can sort of see how your
1:00:28
own problem solving probably makes these mental images of what's going to happen. And the model is able to do that and and
1:00:34
actually render that. Um this is like an example. You can
1:00:39
annotate Orville Wright's airplane with different stats. I guess 605 pounds.
1:00:46
That's interesting. Didn't realize it was quite so heavy.
1:00:51
Um, you know, I I tweeted this out as an example of what it can do. When we
1:00:56
initially launched Nanomana Pro and a bunch of people got very sad.
1:01:02
[laughter] So, no Pluto, where Pluto? So, I said, "Okay, okay.
1:01:08
[snorts] I too am a child of the nine planet era. Um make this image 219 to add Pluto and
1:01:15
add a humorous comment. So it decided to add the quote former planet got demoted
1:01:21
to dwarf planet status still grumpy about it. And he said perfect we are so back.
1:01:28
[laughter] [snorts] Okay, I want to touch a little bit on
1:01:35
how we organize a big effort like Gemini. Uh,
1:01:41
and in particular, um, this is a large effort. I I'm now a co-author on a paper with, you know, more than a thousand
1:01:47
authors. Uh,
1:01:53
our our reports are long. Even just listing the authors takes quite a lot of pages.
1:01:59
[snorts] We always try to spell something funny in the first few words of the of the
1:02:04
author names. Um,
1:02:09
so when you have that many people trying to work together on something, you need a little bit of structure. So we have
1:02:15
overall leads, we have program management and product management that help kind of assess, you know, help us
1:02:22
be organized, but also help us figure out from a product standpoint, what do we want our next generation models to
1:02:27
do? Um and then we have a bunch of different sub areas that are really
1:02:33
really critical to making Gemini work well. So one is you know model development. So in terms of both
1:02:39
pre-training and post-training and RL uh things like that we have a version of
1:02:44
Gemini that runs on device on our pixel phones. So that has somewhat different characteristics and somewhat different
1:02:51
aspects of training data and capabilities needed. Uh in terms of capabilities, we have a bunch of areas
1:02:57
like safety and vision, audio, code, agents, internationalization that we care a lot about. And then core areas
1:03:04
like what data should we train on? How can we get the highest quality data we
1:03:09
can for the token budget we have for training our next model of owls? How do we know if we're doing a good job across
1:03:16
lots of different uh capabilities? Infrastructure, which is like how do we make the training process work well? How
1:03:22
do we make the serving infrastructure work well? Um the codebase and then longer term research that is meant to
1:03:28
feed in good ideas for Gemini N plus2 or Gemini N plus3 that will intersect the
1:03:36
um sort of at the right time. Uh and we have people spread out all
1:03:42
over the world. So we have about a third of the people in in the Bay Area, a third in London and a third in many
1:03:48
other places. So, uh, New York City, Paris, Boston, Zurich, Bangalore. Um,
1:03:53
time zones are super annoying. Not much we can do about it. But the golden hours
1:03:59
between California, West Coast, and London, Europe are there's about three hours a day that are not too terrible
1:04:06
for anyone. Um, and so that's kind of nice. Uh, but that does mean you have to
1:04:12
have asynchronous ways of working, uh, that work well. Uh and so there lots and
1:04:18
lots of large and small discussion groups uh chat rooms, Google chat spaces. So I'm in 200 of these.
1:04:24
[clears throat] Uh we have a request for comment kind of internal document or
1:04:29
tech reporty like thing. Uh which is a semiformal way of getting feedback. You
1:04:35
know, knowing what other people are working on, jotting down an idea. They range in sophistication from early stage
1:04:42
idea I'm thinking about I want comments on to I've done you know an exhaustive
1:04:47
exploration of 10 different variants of this idea which one should we put in the next version of Gemini. They're numbered
1:04:54
sequentially and I just looked we now have more than 5,000 of these. So there's quite a lot of internal requests
1:05:00
for comments that are uh um you know ranging in
1:05:05
uh complexity from a single page thing to a something that would be a tech report or a paper. Um and then
1:05:12
leaderboards and common baselines enable us to make datadriven decisions about how do we improve the model. So we have
1:05:17
lots of rounds of experimentation. you do many many more experiments at small scale and only progress things that seem
1:05:24
promising to to medium and large scale. Um every so often we incorporate say
1:05:30
successful experiments demonstrated at larger scale into a new candidate baseline and then we repeat um and that
1:05:38
seems to have worked pretty well. Uh sorry uh few forward-looking thoughts. I
1:05:45
mean I think one area that's going to be really important is how do humans and AI agents collaborate to get stuff done
1:05:52
right like currently a lot of the uses of these models are you have a single person sitting down with an interactive
1:05:59
you know uh chatbot interface doing stuff but in the future it seems like more work is going to happen where the
1:06:05
human coordinating the activities of a dozen or hundred AI agents doing stuff
1:06:10
on their behalf where they've probably the human is probably weekly specified
1:06:16
what it is they want. How does the you know how can we give um the right HCI
1:06:23
paradigm for managing a team of 50 of these virtual assistants? How can the agents themselves cooperate to
1:06:28
accomplish things? What will this enable? I think these are all pretty interesting uh directions.
1:06:34
Um I think one the million tokens of context is quite useful and that gives
1:06:40
you you know a thousand pages of text or you know hours of video or whatever. Um
1:06:46
but it seems like it'd be even more useful to be able to attend to a trillion tokens of stuff rather than a million. Um, and so I think hybrid
1:06:54
systems with learned retrieval algorithms over large corpora and then maybe lightweight models that can be
1:06:59
used to assess how relevant are these 30,000 documents I retrieved uh to the
1:07:05
thing I'm trying to do and then put the you know maybe hundred things you actually think are really really
1:07:11
important into the context window and maybe premputation from a computer systems uh perspective to make this all
1:07:18
efficient and fast. I think there'd be a lot of uses for this personalized Gemini. I'd love it if I could attend to
1:07:23
all of my email state and photos and so on with my permission of course. Uh web
1:07:30
search, you know, multimodal search and retrieval like searching all the YouTube videos, coding agents, like every coding
1:07:36
agent at Google. It'd be nice if it attended to the whole Google codebase for every Google developer.
1:07:43
Um and then forward-looking, I think inference efficiency is going to be really critical, right? like the the
1:07:48
training cost of these models is very large and you need a lot of chips. But going forward more and more inference is
1:07:54
going to happen and more and more agents using these things in sophisticated ways and interacting with other agents that
1:08:00
themselves are doing inference. So specialized hardware design purely for inference is going to be important.
1:08:06
Again, I think um model and algorithmic improvements for efficient inference are going to be important. Low latency for
1:08:13
these inference cases is a huge plus. It's just much more enjoyable to use something that responds in 100
1:08:19
milliseconds than than 5 seconds. Um, and I I believe in the future that AI
1:08:24
applied to automating chip design is going to be one of the ways in which we get much more efficient inference hardware. I have a whole talk on this,
1:08:31
but essentially it's use learning as much as possible uh and RL. Um, okay.
1:08:38
Uh, I have sorry I'm running over. I have two more slides here. Uh, one of
1:08:44
the other things that we I uh did in the last couple years is we got together a group of co-authors thinking about what
1:08:51
are the impacts of AI in many different domains of the world of the world not just computer science. Um, and we wanted
1:08:59
to like look at what these uh, you know, impacts might be both positive and negative. And so we we identified seven
1:09:07
domains we thought would be pretty dramatically impacted by AI. So employment, education, healthcare,
1:09:12
misinformation, media, entertainment, governance and national security and AI for science. Um and then we went out and
1:09:21
we interviewed a bunch of domain experts in these areas. Uh and in particular,
1:09:26
you know, tried to find people we thought would have interesting perspectives on on one or more of these
1:09:32
uh domain areas. Uh and then we wrote up what we learned from chatting with the
1:09:38
uh the domain experts as well as our own thoughts. Um
1:09:43
and so we produced a long archive paper which uh not that many people have read.
1:09:49
Uh so we have a shorter version in in Kacum and Dave Patterson, one of
1:09:55
the co-authors, wrote a nice one-page editorial on the economist. Um and this is all available on shapingai.com. I
1:10:02
think this is a pretty interesting paper. I think it has some good insights into what will the impact of AI beyond
1:10:07
say employment and technology transitions. Uh as one example, it's the
1:10:13
only paper I have with its own website. Okay. In conclusion, I think AI models
1:10:19
like Gemini and products built with them are these are becoming really important and powerful tools for the things we try
1:10:25
to do every day. Um further research and innovation is going to improve continue this trend. It's going to have a
1:10:32
dramatic impact in many of the areas that I just mentioned in the shaping AI paper, you know, uh, healthcare,
1:10:38
education, science, all kinds of things. Uh, it potentially makes really deep
1:10:44
expertise like these models are really good in some areas and it makes that available to many people all over the
1:10:50
world, which is a pretty powerful uh, capability
1:10:56
and done well, I think our AI assisted future is going to be really bright.
1:11:01
All right. [applause] Thank you.
1:11:20
[clears throat]
1:11:25
question. What's your biggest fear about AI?
1:11:31
My biggest fear of AI? I mean, I think there's a whole range of views on safety of AI systems and so on. You know, I'm
1:11:38
sort of of the view that those are a little overblown. I think like careful engineering of what we allow AI systems
1:11:45
to do will enable us to have like safe deployment of these systems even though they are quite capable. You know, I
1:11:51
think um misinformation is one that I'm worried about in the nearer term because that seems like now we can create
1:11:57
incredibly realistic video and audio uh things that are hard to distinguish even
1:12:04
for super sophisticated people and and with a lot of time to look at them. So that that seems like one area that you
1:12:10
know I I think will be a a worry and I think also just how do we manage uh
1:12:19
you know the impact of AI being able to do things that previously it wasn't able to do and how do we make sure that
1:12:25
people are prepared for those transitions and can learn to use these new tools and can be more productive and
1:12:31
more effective using these tools uh than um you know what could happen which is
1:12:37
you know all of a sudden people suddenly can't can't do that because it's been able automated.
1:12:47
I'm Joey. I'm a senior in computer science. I had a question about hallucinations
1:12:52
and specifically like long behavior especially as we're getting more workflows and like longer contexts. We
1:12:58
know that like even though um past models claim to have one million context window at the limits of that 1 million
1:13:05
context window the retention and like quality of responses um beyond simply
1:13:11
adding search into the language model. What are other ways that Gemini has been priorizing decreasing especially more
1:13:19
like critical [clears throat] applications like yeah I mean I would say one of the key techniques is instead of just generating
1:13:25
a single answer having multiple rollouts of potential answers and then having the
1:13:32
model itself look at those uh rollouts and assess which one it thinks is mo or
1:13:38
which pieces of them it thinks are most likely correct. that that essentially is using much more inference time compute
1:13:45
uh or chain of thought style thinking those kinds of things can be uh you know
1:13:50
definitely decrease the rate of hallucinations and make the I mean that's sort of the kind of approach we
1:13:56
use in some of the IMO work where you really can't afford to hallucinate
1:14:01
because you need to prove this problem this one of these six problems correct um and so that has been quite a quite a
1:14:08
good tool to rely on but that also brings back the question of inference efficiency because for you'd really like
1:14:15
for every answer to produce 16 answers and look at them and decide which ones uh or which which ones you want to throw
1:14:22
out and which ones you want to keep deep keep
1:14:27
I'm curious what's your what's your point of view of LMS versus the emerging world models that are coming up
1:14:35
[clears throat] or being worked on from a research yeah I mean I think uh world models are really just do you understand, you know,
1:14:42
multiple modalities and how the world works from a, you know, a a physics
1:14:48
perspective and and so on. Um, you know, I think a lot of our Gemini models are actually pretty good at at doing world
1:14:55
models. You know, if you think if you look at the uh Genie 3 releases, those
1:15:00
are built on top of Gemini models and enable you to generate uh sort of fairly
1:15:06
sophisticated uh imaginary worlds with text prompts and then control interaction in that
1:15:14
world. So you can move left, move right, jump, move forward and move back. And it maintains consistency with the the world
1:15:21
that it has created. And if you turn left and then turn back right, the things that were there are still there.
1:15:27
Um, so that's an important thing. You know, we're actually collaborating with our Whimo colleagues because that enables them to create longtail training
1:15:35
cases in simulation. So you can say, please make a scenario where an elephant
1:15:40
appears in the middle of the road and that enables you to like create, you
1:15:46
know, dozens and dozens of test cases that are really hard to actually have occur in the real world. And you can see
1:15:51
how the car behaves and it actually even synthesizes the LAR data that you the
1:15:56
the system would would would have enabling you to get better uh you know safety properties.
1:16:04
Maybe one more question Jeff you can [laughter] pressure all right not high
1:16:12
not low but middle. Hi. Um, I guess like the last question
1:16:17
research. I'm kind of curious what kind of what type of research you think we should work on about a thousand TPUs
1:16:24
because a few of the directions we talked about are like pretty data inensive. Yeah. What do you think we can do to help the
1:16:30
AI forward? Yeah. I mean, I think a lot of the kinds of work we do from a research
1:16:36
perspective start as investigations at very small scale, right? And so I think
1:16:42
where you can be quite effective is trying lots of different things that uh
1:16:49
you maybe you're not training on a thousand accelerators but you're demonstrating at small scale this idea
1:16:55
has legs right like there's all kinds of problems in say continual learning or
1:17:02
different model architectures where I think at smallcale and in particular one of the things we do is we run extremely
1:17:09
extremely small, extremely small, and small. And we look at the trend of those
1:17:14
things because that trend is often more indicative of importance than exactly
1:17:19
where it falls relative to the current baseline state-of-the-art of our extra
1:17:25
extra small or extra small or small model. Right? If the slope, if it's below the baseline, but the slope looks
1:17:31
good, that's a really interesting idea. If it's above the baseline at the absolute smallest scale but rapidly
1:17:37
plummeting below the baseline even at the the the ex the small scale that's
1:17:43
less interesting and so I think where people who don't have access to large
1:17:48
scale compute can really do a great job is focusing on problem uh you know
1:17:56
solutions that are quite different and quite interesting and not stress about
1:18:01
getting state-of-the-art results and I think we as a community also should say this is very different and unique and
1:18:08
doesn't achieve state-of-the-art results but it looks interesting at small scale like we should as you know conference
1:18:14
reviewers or whatever celebrate that kind of thing than something that gets an incremental improvement on
1:18:20
state-of-the-art by tweaking one little thing that is actually not that interesting
1:18:25
question okay [applause] thank Thank you.
