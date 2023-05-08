package main

type YourModel struct {
	// TODO: replace these with real attributes
	PolicyNet interface{}
	Device    string
}

// New creates a new YourModel instance with fields that must be initialised
// fresh at every run.
func New(device string) *YourModel {
	return &YourModel{
		Device: device,
	}
}

func (m *YourModel) FromFile(filename string) *YourModel {
	// todo: open file to get policy net
	policyNet := "replace this with stuff read from the file"
	m.PolicyNet = policyNet

	return m
}

func (m *YourModel) FromFreshInstance(policyNet interface{}) *YourModel {
	m.PolicyNet = policyNet

	return m
}
