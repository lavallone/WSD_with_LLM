shortcut_model_name2full_model_name = {"llama_1b" : "meta-llama/Llama-3.2-1B-Instruct",
                                       "gemma_2b" : "google/gemma-2-2b-it",
                                       "llama_3b" : "meta-llama/Llama-3.2-3B-Instruct",
                                       "phi_mini" : "microsoft/Phi-3.5-mini-instruct",
                                       "phi_small" : "microsoft/Phi-3-small-128k-instruct",
                                       "mistral2" : "mistralai/Mistral-7B-Instruct-v0.3",
                                       "mistral" : "mistralai/Ministral-8B-Instruct-2410",
                                       "llama_8b" : "meta-llama/Llama-3.1-8B-Instruct",
                                       "gemma_9b" : "google/gemma-2-9b-it",
                                       "gpt" : "gpt-4o"}


chat_template_prompts = {
                            "selection":{
                                        "zero_shot": {
                                                        "prompt" : """Consider the following sentence: {text}\n\nSelect the most suitable dictionary definition which identifies the meaning of "{word}" among the following definitions:\n\n{candidate_definitions}"""
                                        },
                                        "one_shot": {
                                                        "example" : """Consider the following sentence: they needed rugs to cover the bare floors.\n\nSelect the most suitable dictionary definition which identifies the meaning of "floor" among the following definitions:\n\n1) the inside lower horizontal surface (as of a room, hallway, tent, or other structure).\n2) a structure consisting of a room or set of rooms at a single position along a vertical scale.\n3) a lower limit.\n4) the ground on which people and animals move about.\n5) the bottom surface of any lake or other body of water.\n6) the lower inside surface of any hollow structure.\n7) the occupants of a floor.\n8) the parliamentary right to address an assembly.\n9) the legislative hall where members debate and vote and conduct other business.\n10) a large room in a exchange where the trading is done.\n""", 
                                                        "example_output" : """1) the inside lower horizontal surface (as of a room, hallway, tent, or other structure).""", 
                                                        "prompt" : """Consider the following sentence: {text}\n\nSelect the most suitable dictionary definition which identifies the meaning of "{word}" among the following definitions:\n\n{candidate_definitions}"""
                                        },
                                        "few_shot": {
                                                        "example_1" : """Consider the following sentence: I am using a computer for the creation of 3-D models.\n\nSelect the most suitable dictionary definition which identifies the meaning of "computer" among the following definitions:\n\n1) a machine for performing calculations automatically.\n2) an expert at calculation (or at operating calculating machines).\n""", 
                                                        "example_1_output" : """1) a machine for performing calculations automatically.""", 
                                                        "example_2" : """Consider the following sentence: a great delay.\n\nSelect the most suitable dictionary definition which identifies the meaning of "great" among the following definitions:\n\n1) relatively large in size or number or extent; larger than others of its kind.\n2) of major significance or importance.\n3) remarkable or out of the ordinary in degree or magnitude or effect.\n4) very good.\n5) uppercase.\n6) in an advanced stage of pregnancy.\n""", 
                                                        "example_2_output" : """1) relatively large in size or number or extent; larger than others of its kind.""", 
                                                        "example_3" : """Consider the following sentence: What did you eat for dinner last night?\n\nSelect the most suitable dictionary definition which identifies the meaning of "eat" among the following definitions:\n\n1) take in solid food.\n2) eat a meal; take a meal.\n3) take in food; used of animals only.\n4) worry or cause anxiety in a persistent way.\n5) cause to deteriorate due to the action of water, air, or an acid.\n""", 
                                                        "example_3_output" : """1) take in solid food.""", 
                                                        "prompt" : """Consider the following sentence: {text}\n\nSelect the most suitable dictionary definition which identifies the meaning of "{word}" among the following definitions:\n\n{candidate_definitions}"""
                                        },
                            },
                            "generation":{
                                        "zero_shot": {
                                                        "prompt" : """Consider the following sentence: {text}\n\nProvide a dictionary definition which identifies the meaning of "{word}" in the above sentence and do not motivate your answer."""
                                        },
                                        "one_shot": {
                                                        "example" : """Consider the following sentence: they needed rugs to cover the bare floors.\n\nProvide a dictionary definition which identifies the meaning of "floor" in the above sentence and do not motivate your answer.""", 
                                                        "example_output" : """The inside lower horizontal surface (as of a room, hallway, tent, or other structure).""", 
                                                        "prompt" : """Consider the following sentence: {text}\n\nProvide a dictionary definition which identifies the meaning of "{word}" in the above sentence and do not motivate your answer."""
                                        },
                                        "few_shot": {
                                                        "example_1" : """Consider the following sentence: I am using a computer for the creation of 3-D models.\n\nProvide a dictionary definition which identifies the meaning of "computer" in the above sentence and do not motivate your answer.""", 
                                                        "example_1_output" : """A machine for performing calculations automatically.""", 
                                                        "example_2" : """Consider the following sentence: a great delay.\n\nProvide a dictionary definition which identifies the meaning of "great" in the above sentence and do not motivate your answer.""", 
                                                        "example_2_output" : """Relatively large in size or number or extent; larger than others of its kind.""", 
                                                        "example_3" : """Consider the following sentence: What did you eat for dinner last night?\n\nProvide a dictionary definition which identifies the meaning of "eat" in the above sentence and do not motivate your answer.""", 
                                                        "example_3_output" : """Take in solid food.""", 
                                                        "prompt" : """Consider the following sentence: {text}\n\nProvide a dictionary definition which identifies the meaning of "{word}" in the above sentence and do not motivate your answer."""
                                        },   
                            },
}


# OLD PROMPT VERSIONS
# chat_template_prompts = {
#                             "selection":{
#                                         "zero_shot": {
#                                                         "prompt" : """Select the most suitable meaning for '{word}' in the following sentence: {text}\nChoose the corresponding definition among: \n{candidate_definitions}.\nAnswer by reporting the corresponding definition and do not motivate your answer."""
#                                         },
#                                         "one_shot": {
#                                                         "example" : """Select the most suitable meaning for "floor" in the following sentence: they needed rugs to cover the bare floors. Choose the corresponding definition among: \n1) the inside lower horizontal surface (as of a room, hallway, tent, or other structure).\n2) a structure consisting of a room or set of rooms at a single position along a vertical scale.\n3) a lower limit.\n4) the ground on which people and animals move about.\n5) the bottom surface of any lake or other body of water.\n6) the lower inside surface of any hollow structure.\n7) the occupants of a floor.\n8) the parliamentary right to address an assembly.\n9) the legislative hall where members debate and vote and conduct other business.\n10) a large room in a exchange where the trading is done.\nAnswer by reporting the corresponding definition and do not motivate your answer.""", 
#                                                         "example_output" : """1) the inside lower horizontal surface (as of a room, hallway, tent, or other structure).""", 
#                                                         "prompt" : """Select the most suitable meaning for '{word}' in the following sentence: {text}\nChoose the corresponding definition among: \n{candidate_definitions}.\nAnswer by reporting the corresponding definition and do not motivate your answer."""
#                                         },
#                                         "few_shot": {
#                                                         "example_1" : """Select the most suitable meaning for "computer" in the following sentence: I am using a computer for the creation of 3-D models. Choose the corresponding definition among: \n1) a machine for performing calculations automatically.\n2) an expert at calculation (or at operating calculating machines).\nAnswer by reporting the corresponding definition and do not motivate your answer.""", 
#                                                         "example_1_output" : """1) a machine for performing calculations automatically.""", 
#                                                         "example_2" : """Select the most suitable meaning for "great" in the following sentence: a great delay. Choose the corresponding definition among: \n1) relatively large in size or number or extent; larger than others of its kind.\n2) of major significance or importance.\n3) remarkable or out of the ordinary in degree or magnitude or effect.\n4) very good.\n5) uppercase.\n6) in an advanced stage of pregnancy.\nAnswer by reporting the corresponding definition and do not motivate your answer.""", 
#                                                         "example_2_output" : """1) relatively large in size or number or extent; larger than others of its kind.""", 
#                                                         "example_3" : """Select the most suitable meaning for "eat" in the following sentence: What did you eat for dinner last night? Choose the corresponding definition among: \n1) take in solid food.\n2) eat a meal; take a meal.\n3) take in food; used of animals only.\n4) worry or cause anxiety in a persistent way.\n5) cause to deteriorate due to the action of water, air, or an acid.\nAnswer by reporting the corresponding definition and do not motivate your answer.""", 
#                                                         "example_3_output" : """1) take in solid food.""", 
#                                                         "prompt" : """Select the most suitable meaning for '{word}' in the following sentence: {text}\nChoose the corresponding definition among: \n{candidate_definitions}.\nAnswer by reporting the corresponding definition and do not motivate your answer."""
#                                         },
#                             },
#                             "generation":{
#                                         "zero_shot": {
#                                                         "prompt" : """Define the meaning of "{word}" in the following sentence: {text} Provide a definition which identifies the meaning of "{word}" in the context provided and do not motivate your answer."""
#                                         },
#                                         "one_shot": {
#                                                         "example" : """Define the meaning of "floor" in the following sentence: they needed rugs to cover the bare floors. Provide a definition which identifies the meaning of "floor" in the context provided and do not motivate your answer.""", 
#                                                         "example_output" : """The inside lower horizontal surface (as of a room, hallway, tent, or other structure).""", 
#                                                         "prompt" : """Define the meaning of "{word}" in the following sentence: {text} Provide a definition which identifies the meaning of "{word}" in the context provided and do not motivate your answer."""
#                                         },
#                                         "few_shot": {
#                                                         "example_1" : """Define the meaning of "computer" in the following sentence: I am using a computer for the creation of 3-D models. Provide a definition which identifies the meaning of "computer" in the context provided and do not motivate your answer.""", 
#                                                         "example_1_output" : """A machine for performing calculations automatically.""", 
#                                                         "example_2" : """Define the meaning of "great" in the following sentence: a great delay. Provide a definition which identifies the meaning of "great" in the context provided and do not motivate your answer.""", 
#                                                         "example_2_output" : """Relatively large in size or number or extent; larger than others of its kind.""", 
#                                                         "example_3" : """Define the meaning of "eat" in the following sentence: What did you eat for dinner last night? Provide a definition which identifies the meaning of "eat" in the context provided and do not motivate your answer.""", 
#                                                         "example_3_output" : """Take in solid food.""", 
#                                                         "prompt" : """Define the meaning of "{word}" in the following sentence: {text} Provide a definition which identifies the meaning of "{word}" in the context provided and do not motivate your answer."""
#                                         },   
#                             },
# }