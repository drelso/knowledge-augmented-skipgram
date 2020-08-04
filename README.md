# On the Effects of Knowledge-Augmented Data in Word Embeddings

## Description

These files provide an unpolished version (i.e. not optimised for ease of use) of the code used to produce the results in the accompanying paper. Due to space restrictions, data and model files could not be provided, so these files are meant to present the implementation details for the paper, but full reproduction of the results will only be available after the anonymity period.

The two main files are `train_skipgram.py` to train the Skip-gram model and save the learned embeddings, and `wmd-relax.py` to perform multi-processed K-Nearest Neighbour Word Mover's Distance document classification.


## Train set books

0. R M Ballantyne___The Madman and the Pirate.txt
1. James Fenimore Cooper___The Last of the Mohicans.txt
2. James Otis___On the Kentucky Frontier.txt
3. Louisa May Alcott___Under the Lilacs.txt
4. Daniel Defoe___The Further Adventures of Robinson Crusoe.txt
6. Hector Hugh Munro___When William Came.txt
7. Mark Twain___A Connecticut Yankee in King Arthur's Court, Complete.txt
8. R M Ballantyne___Six Months at the Cape.txt
9. Herbert George Wells___The Time Machine.txt
10. Jane Austen___Mansfield Park.txt
11. Charles Kingsley___True Words for Brave Men.txt
13. R M Ballantyne___The Life of a Ship.txt
14. Bret Harte___Mrs. Skaggs's Husbands and Other Stories.txt
15. G K Chesterton___Robert Browning.txt
16. Daniel Defoe___Second Thoughts are Best.txt
17. William Penn___A Brief Account of the Rise and Progress of the People Called Quakers.txt
18. Andrew Lang___Historical Mysteries.txt
19. P G Wodehouse___The Prince and Betty.txt
20. R M Ballantyne___The Lively Poll.txt
21. William Dean Howells___Literary Boston.txt
22. Winston Churchill___A Modern Chronicle, Complete.txt
23. Lyman Frank Baum___Aunt Jane's Nieces at Millville.txt
24. Jacob Abbott___Mary Erskine.txt
25. Howard Pyle___The Rose of Paradise.txt
26. Mark Twain___The Facts Concerning The Recent Carnival Of Crime In Connecticut.txt
27. George Alfred Henty___The Cornet of Horse.txt
28. Sir Walter Scott___The Dramatic Works of John Dryden, Volume I.txt
29. William Makepeace Thackeray___The Wolves and the Lamb.txt
30. George Bernard Shaw___An Unsocial Socialist.txt
31. Jonathan Swift___The Prose Works of Jonathan Swift, D.D., Volume 4.txt
32. Nathaniel Hawthorne___Passages from the French and Italian Notebooks, Complete.txt
33. Jacob Abbott___Cleopatra.txt
34. Henry James___Notes of a Son and Brother.txt
35. Edward Phillips Oppenheim___The Pawns Count.txt
36. Walt Whitman___Leaves of Grass.txt
37. George Alfred Henty___On the Irrawaddy.txt
38. Walter de la Mare___Henry Brocken.txt
39. Frank Richard Stockton___The Great Stone of Sardis.txt
40. Walter de la Mare___Collected Poems 1901-1918 in Two Volumes Volume 1.txt
41. William Wordsworth___The Poetical Works of William Wordsworth, Volume 4.txt
42. Bret Harte___Stories in Light and Shadow.txt
43. Anthony Trollope___The Man Who Kept His Money In A Box.txt
44. Lewis Carroll___Phantasmagoria and Other Poems.txt
45. William Makepeace Thackeray___The Bedford-Row Conspiracy.txt
46. William Dean Howells___A Traveler from Altruria.txt
47. John Ruskin___Mornings in Florence.txt
48. Joseph Conrad___To-morrow.txt
50. Frank Richard Stockton___The Great Stone of Sardis.txt
51. Sir Arthur Conan Doyle___The Green Flag.txt
52. P G Wodehouse___My Man Jeeves.txt
53. Henry James___The Diary of a Man of Fifty.txt
54. Henry David Thoreau___On the Duty of Civil Disobedience.txt
55. Baronness Orczy___The Old Man in the Corner.txt
56. Rafael Sabatini___Love-at-Arms.txt
57. Lyman Frank Baum___The Road to Oz.txt
58. John Galsworthy___The Little Dream.txt
59. Oscar Wilde___Miscellanies.txt
60. William Dean Howells___The Story of a Play.txt
62. Rudyard Kipling___The Day's Work, Volume 1.txt
63. Abraham Lincoln___The Writings of Abraham Lincoln, Volume 6: 1862-1863.txt
64. John Galsworthy___The Silver Box.txt
65. Jerome Klapka Jerome___Diary of a Pilgrimage.txt
66. Isaac Asimov___Youth.txt
67. R M Ballantyne___The Crew of the Water Wagtail.txt
68. Edward Phillips Oppenheim___The Vanished Messenger.txt
69. Jack London___The Road.txt
70. Daniel Defoe___Atalantis Major.txt
71. Charles Darwin___The Different Forms of Flowers on Plants of the Same Species.txt
72. Sir Arthur Conan Doyle___Memoirs of Sherlock Holmes.txt
73. Sir Arthur Conan Doyle___The Mystery of Cloomber.txt
74. William Wymark Jacobs___Good Intentions, Ship's Company, Part 3.txt
75. G K Chesterton___Wine, Water, and Song.txt
76. Beatrix Potter___The Tale Of Peter Rabbit.txt
77. Benjamin Disraeli___Venetia.txt
78. William Wymark Jacobs___Odd Craft.txt
79. Abraham Lincoln___The Writings of Abraham Lincoln, Volume 7: 1863-1865.txt
80. Robert Louis Stevenson___Edinburgh.txt
81. Edward Phillips Oppenheim___The Pawns Count.txt
82. Charles Dickens___American Notes for General Circulation.txt
83. Ralph Waldo Emerson___Essays, Second Series.txt
84. Joseph Conrad___Typhoon.txt
86. William Wymark Jacobs___Friends In Need, Ship's Company, Part 2.txt
87. Abraham Lincoln___Lincoln's Gettysburg Address, given November 19, 1863.txt
88. Walter de la Mare___Peacock Pie, A Book of Rhymes.txt
89. Thornton Waldo Burgess___Mother West Wind "Where" Stories.txt
90. Jacob Abbott___Rollo on the Atlantic.txt
91. Henry Rider Haggard___The Brethren.txt
92. Jack London___Michael, Brother of Jerry.txt
93. William Wymark Jacobs___Dixon's Return, Odd Craft, Part 10.txt
94. John Keats___Lamia.txt
95. Robert Louis Stevenson___The Works of Robert Louis Stevenson - Swanston Edition, Volume 21.txt
96. Stephen Leacock___Frenzied Fiction.txt
97. Edward Stratemeyer___Leo the Circus Boy.txt
98. Winston Churchill___The Dwelling Place of Light, Complete.txt
99. Thomas Carlyle___History of Friedrich II of Prussia, Volume 7.txt
100. Herbert George Wells___The Food of the Gods and How It Came to Earth.txt
101. Charles Dickens___Mrs. Lirriper's Legacy.txt
102. Joseph Conrad___Tales of Unrest.txt
103. Charles Dickens___Speeches: Literary and Social.txt
104. Abraham Lincoln___Lincoln's Second Inaugural Address.txt
105. Joseph Conrad___The Arrow of Gold.txt
106. Benjamin Disraeli___The Young Duke.txt
107. Henry Rider Haggard___The Yellow God.txt
108. William Wymark Jacobs___Made to Measure, Deep Waters, Part 3.txt
109. Edward Stratemeyer___Four Boy Hunters.txt
110. William Wordsworth___Lyrical Ballads, With Other Poems, 1800, Volume 1.txt
111. Lyman Frank Baum___Policeman Bluejay.txt
112. Edward Stratemeyer___The Rover Boys on Snowshoe Island.txt
114. George Alfred Henty___A Search For A Secret, a Novel, Volume 3.txt
115. Robert Louis Stevenson___A Christmas Sermon.txt
116. Alfred Russel Wallace___The Malay Archipelago, Volume 2.txt
117. George Alfred Henty___The Young Carthaginian.txt
118. John Milton___Milton's Comus.txt
119. Harold Bindloss___Kit Musgrave's Luck.txt
120. Charles Kingsley___Westminster Sermons.txt
121. Henry Rider Haggard___Pearl-Maiden.txt
122. Albert Einstein___Relativity: The Special and General Theory.txt
123. John Galsworthy___Beyond.txt
124. Ambrose Bierce___Write It Right.txt
125. Sir Richard Francis Burton___Two Trips to Gorilla Land and the Cataracts of the Congo, Volume 1.txt
126. Lucy Maud Montgomery___Anne Of The Island.txt
127. Nathaniel Hawthorne___Fancy's Show-Box (From "Twice Told Tales").txt
128. John Ruskin___The Seven Lamps of Architecture.txt
129. William Wymark Jacobs___A Spirit of Avarice, Odd Craft, Part 11.txt
130. John Galsworthy___Six Short Plays, Complete.txt
131. Sinclair Lewis___Babbitt.txt
133. James Joyce___Chamber Music.txt
134. Jacob Abbott___Alexander the Great.txt
135. Herbert George Wells___Mr. Britling Sees It Through.txt
136. R M Ballantyne___The Middy and the Moors.txt
137. John Galsworthy___Joy.txt
138. Thomas Hardy___The Romantic Adventures of a Milkmaid.txt
139. Bret Harte___By Shore and Sedge.txt
140. Wilkie Collins___Basil.txt
141. James Matthew Barrie___Peter Pan in Kensington Gardens, Version 1.txt
142. Benjamin Disraeli___Coningsby.txt
143. Thomas Henry Huxley___Note on the Resemblances and Differences in the Structure and the Development of Brain in Man and the Apes.txt
144. Howard Pyle___Men of Iron.txt
145. William Makepeace Thackeray___The Rose and the Ring.txt
146. William Wymark Jacobs___Sea Urchins.txt
147. Charles Dickens___Some Christmas Stories.txt
148. Winston Churchill___A Far Country, Complete.txt
149. William Wymark Jacobs___A Master of Craft.txt
150. William Wymark Jacobs___The Understudy, Night Watches, Part 3.txt
151. Harold Bindloss___Blake's Burden.txt
152. Herbert George Wells___The Red Room.txt
153. Bret Harte___The Three Partners.txt
154. Jerome Klapka Jerome___Idle Thoughts of an Idle Fellow.txt
155. Oscar Wilde___Miscellanies.txt
156. Herbert George Wells___Floor Games; a companion volume to "Little Wars".txt
157. Lyman Frank Baum___The Wonderful Wizard of Oz.txt
158. Andrew Lang___Much Darker Days.txt
159. Charles Kingsley___Sermons for the Times.txt


## Validation set books

1. Lewis Carroll___Rhyme? And Reason?.txt
2. Baronness Orczy___The Tangled Skein.txt
3. James Matthew Barrie___What Every Woman Knows.txt
4. Jack London___Dutch Courage and Other Stories.txt
5. George Bernard Shaw___Mrs. Warren's Profession.txt
6. Philip Kindred Dick___Mr. Spaceship.txt
7. Beatrix Potter___The Tale of Ginger and Pickles.txt
8. Herbert Spencer___The Philosophy of Style.txt
9. Edmund Burke___The Works of the Right Honourable Edmund Burke, Vol. 01 (of 12).txt
10. Elizabeth Barrett Browning___The Poetical Works of Elizabeth Barrett Browning Volume IV.txt