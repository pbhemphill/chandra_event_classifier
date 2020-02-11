# chandra_event_classifier

Uses a TensorFlow/Keras neural network to classify events in unfiltered Chandra
eventlists as good or bad. Not a substitute for the actual Chandra pipeline,
which works better, is faster, and actually does things right (as in, based on
in-depth testing and calibration and actual physics).

Currently only tested on a single LETG-HRC observation, so my mileage may vary.
But gets about 95% correct on that one dataset, so that's nice.

TODO:
- Train and test with event files from different configurations and detectors
