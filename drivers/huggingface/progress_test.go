package huggingface

import (
	"math"
	"strings"
	"testing"

	"github.com/OpenListTeam/OpenList/v4/internal/model"
)

// TestProgressSpoolContract pins progressReader to the driver contract:
// 0-100 percentage points, reaching exactly 100 when the stream is drained.
func TestProgressSpoolContract(t *testing.T) {
	var last float64
	pr := &progressReader{
		r:    strings.NewReader("0123456789"),
		size: 10,
		up:   func(p float64) { last = p },
	}
	buf := make([]byte, 3)
	for {
		if _, err := pr.Read(buf); err != nil {
			break
		}
	}
	if last < 0 || last > 100 || math.Abs(last-100) > 1e-9 {
		t.Fatalf("expected progress 100, got %v", last)
	}
}

// TestProgressRangeMaps checks UpdateProgressWithRange maps the input
// (0-100) onto the spool share of the bar.
func TestProgressRangeMaps(t *testing.T) {
	var last float64
	up := func(p float64) { last = p }
	rp := model.UpdateProgressWithRange(up, 0, 60)
	rp(0)
	first := last
	rp(50)
	mid := last
	rp(100)
	final := last
	if first != 0 || math.Abs(mid-30) > 1e-9 || final != 60 {
		t.Fatalf("bad range mapping: %v %v %v", first, mid, final)
	}
}

// TestProgressMultipartFlow mirrors lfsUploadMultipart's per-part callback:
// one callback per part over the 60-100 share, monotonic, ending at 100.
func TestProgressMultipartFlow(t *testing.T) {
	events := []float64{}
	up := func(p float64) { events = append(events, p) }
	size := int64(96)
	ends := []int64{16, 32, 48, 64, 80, 96} // 6 parts of a 96-byte file
	for _, end := range ends {
		model.UpdateProgressWithRange(up, 60, 100)(float64(end) / float64(size) * 100)
	}
	if len(events) != len(ends) {
		t.Fatalf("expected %d callbacks, got %d", len(ends), len(events))
	}
	for i := 1; i < len(events); i++ {
		if events[i] < events[i-1] {
			t.Fatalf("progress went backwards: %v", events)
		}
	}
	// first part already moved 1/6 of the file: 60 + 40*(16/96) = 66.67
	if math.Abs(events[0]-200.0/3.0) > 1e-9 || math.Abs(events[len(events)-1]-100) > 1e-9 {
		t.Fatalf("expected first=66.67 last=100, got %v", events)
	}
}
