# Spatial Reasoning Error Taxonomy

A predefined taxonomy of expected failure modes for transformer-based
spatial reasoning models on SpartQA-YN. Categories are mutually exclusive;
when an error fits multiple, pick the most specific.

## Categories

### 1. Axis confusion
The model treats one spatial axis as if it were a different orthogonal axis.
For example, treating "above" as "left of," or interpreting "to the right of"
as a vertical relation. Indicates the model has not learned the distinction
between the projective frames of reference.

### 2. Relation polarity error
The model gets the spatial axis correct but inverts the direction.
For example, predicting "above" when the correct relation is "below," or
"left of" when the correct relation is "right of." This is a more localized
failure than axis confusion.

### 3. Topological-projective confusion
The model confuses containment or proximity relations (in, inside, contains,
near, touching) with directional/projective ones (above, below, left, right).
Suggests the model has merged distinct spatial relation types into one
representation.

### 4. Multi-hop composition failure
The question requires chaining two or more spatial relations across the
story to arrive at the answer (e.g., "A is left of B, and B is above C,
so A is to the upper-left of C"). The model gets simple single-hop
relations correct but fails when reasoning must compose.

### 5. Distractor object error
When the story contains multiple objects sharing some attribute (color,
shape, size), the model attends to the wrong entity. The spatial relation
itself may be reasoned correctly with respect to the wrong object.

### 6. DK-vs-decision error
The model commits to "Yes" or "No" when the correct answer is "Don't Know"
(the story does not provide enough information to decide), or vice versa.
This category probes whether the model recognizes the difference between
"the relation does not hold" and "the relation cannot be determined."

### 7. Other / uncategorized
The error does not fit cleanly into any of the above. Brief free-text
description provided per case.

## Annotation procedure

1. After training, sort the test split errors by question difficulty
   (longest stories first as a rough proxy).
2. Sample 100 errors uniformly from the resulting list.
3. For each error, read story + question + gold + pred. Assign exactly
   one category.
4. Compute the distribution. Note any subcategory patterns within
   "Other" that might warrant adding to the taxonomy in future work.

## Inter-annotator reliability

Single-annotator analysis for this project. Reliability would be a natural
extension if a second pass over the data is performed.
