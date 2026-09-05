---
title: "1.6 The Bitter Lesson"
chapter_title: "Fundamentals of Deep Learning"
section_id: "01-06"
language: en
source_language: zh
source_docx: "第1部分 深度学习/1.深度学习基础理论/1.6 The Bitter Lesson.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 1.6 The Bitter Lesson

Richard Sutton published *The Bitter Lesson* in 2019. Its central argument is that general methods based on search and learning will ultimately overwhelm methods designed around human knowledge as computation scales. This anticipated the development of large models and continues to be borne out today. The Chinese edition presents a Chinese translation of the original essay below.

The most important lesson from the past 70 years of artificial intelligence research is that general methods leveraging computation ultimately prove most effective, and by an overwhelming margin. The underlying reason is Moore's law, or, more precisely, the continuing exponential decline in the cost of computation. Most AI research implicitly assumes that the computation available to an agent is fixed (under this assumption, incorporating human knowledge is almost the only way to improve performance). But extending the time horizon even slightly beyond a typical research project reveals increases in computation by orders of magnitude. Researchers tend to exploit their understanding of a domain to obtain results in the short term. In the long run, however, only one thing truly matters: how to exploit computation. In theory these two approaches can proceed together, but in practice they often exclude each other. Time spent on one cannot be spent on the other. Psychological path dependence also develops. More troublingly, human-knowledge-oriented methods often make systems so complex that they hinder the computational advantages of general methods. AI researchers have repeatedly learned this bitter lesson too late. Reviewing some of the most characteristic cases is instructive.

In chess, the method that defeated Kasparov in 1997 relied fundamentally on large-scale, deep search. Most computer-chess researchers at the time were unhappy about this. They had been working on exploiting human understanding of the structure of chess positions. When a simpler, search-based method coupled with specialized hardware and software proved far more effective, these researchers did not lose gracefully. They argued that brute-force search had won this time, but that it was not a general strategy and was not how humans played chess. They wanted a human-knowledge-based method to win; it did not, and they were disappointed.

A similar development occurred in Go, only 20 years later. Early efforts invested heavily in avoiding search and finding ways to exploit human knowledge and the special structure of Go. Once search was applied effectively at scale, all those efforts became irrelevant or even counterproductive. Equally important was learning a value function through self-play (learning to judge the quality of positions, an approach that is crucial in many games, including chess, although learning did not play a major role in the program that first defeated the world champion in 1997). Learning through self-play, and learning itself, are ways of putting large amounts of computation to work, just as search is. Search and learning are the two most important classes of techniques for exploiting vast amounts of computation in AI research. In Go, as in chess, researchers initially concentrated on exploiting human understanding (thereby reducing search), and only much later achieved far greater success by embracing search and learning.

In speech recognition, DARPA funded an early competition in the 1970s. Many entrants used specialized methods that exploited human knowledge about words, phonemes, the human vocal tract, and more. On the other side were emerging statistical methods based on hidden Markov models (HMMs), which required more computation. Statistical methods again defeated human-knowledge-oriented approaches. This initiated a decades-long gradual shift throughout natural language processing, with statistics and computation coming to dominate the field. The recent rise of deep learning in speech recognition is the latest step in this direction. Deep learning methods depend less on human knowledge, use more computation, and learn from enormous training sets, producing much better speech recognition systems. As in games, researchers repeatedly tried to make systems operate in the way they thought their own brains worked. They tried to incorporate this knowledge into their systems. Ultimately, this proved counterproductive and a tremendous waste of researchers' time, because Moore's law made vast amounts of computation available and we found ways to exploit it.

The same happened in computer vision. Early approaches understood vision as searching for edges, generalized cylinders, or SIFT features. Today, these have all been abandoned. Modern deep learning neural networks use only convolution and certain notions of invariance, yet perform much better.

This is a major lesson. As a field, we have still not fully learned it, because we continue to make the same mistakes. To see this clearly and avoid it effectively, we must understand why these mistakes are attractive. We must learn the bitter lesson: embedding what we think is our way of thinking into systems does not work in the long run.

The bitter lesson rests on the following historical observations: 1) AI researchers often try to embed knowledge in agents; 2) this always helps in the short term and gives the researchers a strong personal sense of achievement; 3) in the long run, however, it reaches a bottleneck and may even obstruct further progress; 4) breakthroughs eventually come from the opposite approach, scaling computation through search and learning. The eventual success is tinged with bitterness and is often never fully accepted, because it surpasses the favored, human-centered approach.

The first thing to learn from the bitter lesson is the immense power of general methods. These methods can continue to scale as computation increases, even when it becomes extremely large. Search and learning are two methods capable of scaling without limit in this way.

The second is that the actual contents of the human mind are extraordinarily complex, and that complexity cannot be simplified. We should stop looking for simple ways to think about the contents of the mind, such as simple concepts of space, objects, multiple agents, or symmetries. All of these are parts of the external world, and that world is arbitrary and inherently complex. We should not embed these contents in systems, because their complexity is endless. We should embed only meta-methods that can discover and capture such arbitrary complexity. The key to these methods is that they can find good approximations, but the search should be performed by our methods rather than by us humans. We want agents that can discover as we do, rather than systems that incorporate what we have already discovered. Building systems around our existing discoveries only makes it harder to see how the process of discovery itself works.

## References

- Sutton, R. S. (2019). [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html).
