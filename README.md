# FlakyClassifier

This is the code for our study --- A Large Language Model Approach to Identify
Flakiness in C++ Projects.

# Abstract

The role of regression testing in software testing is crucial as it ensures that any new modifications do not disrupt the existing functionality and behaviour of the software system.
The desired outcome is for regression tests to yield identical
results without any modifications made to the system being tested.
In practice, however, the presence of Flaky Tests introduces
non-deterministic behaviour and undermines the reliability of
regression testing results.
In this paper, we propose an LLM-based approach for
identifying the root cause of flaky tests in C++ projects at
the code level, with the intention of assisting developers in
debugging and resolving them more efficiently. We compile a
comprehensive collection of C++ project flaky tests sourced from
GitHub repositories. We fine-tune Mistral-7b, Llama2-7b and
CodeLlama-7b models on the C++ dataset and an existing Java
dataset and evaluate the performance in terms of precision,
recall, accuracy, and F1 score. We assess the performance of
the models across various datasets and offer recommendations
for both research and industry applications.
The results indicate that our models exhibit varying performance
on the C++ dataset, while their performance is comparable
to that of the Java dataset. The Mistral-7b surpasses the other two
models regarding all metrics, achieving a score of 1. Our results
demonstrate the exceptional capability of LLMs to accurately
classify flakiness in C++ and Java projects, providing a promising
approach to enhance the efficiency of debugging flaky tests in
practice

