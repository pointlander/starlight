// Copyright 2024 The Starlight Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"

	"github.com/pointlander/compress"
	"github.com/pointlander/datum/iris"
	"github.com/pointlander/matrix"
	"github.com/pointlander/pagerank"
	"github.com/pointlander/starlight/kmeans"
)

const (
	// Clusters is the number of clusters
	Clusters = 4
)

// Vector is a vector with labels
type Vector struct {
	Number int
	Vector []float64
	Labels []uint8
}

// Vectors is a set of vectors
type Vectors struct {
	Width   int
	Vectors []Vector
	Rng     *rand.Rand
	Mean    []float64
	Stddev  []float64
}

// Size is the size of vectors
func (v *Vectors) Size() int {
	return v.Width * len(v.Vectors)
}

// Sort sorts vectors by a column
func (v *Vectors) Sort(col int) {
	s := Sorter{
		Vec: v,
		Col: col,
	}
	sort.Sort(s)
}

// Bounds are the range over which to compute the split
type Bounds struct {
	Begin int
	End   int
}

// Split the split point
type Split struct {
	Col   int
	Index int
	Var   float64
}

func (v *Vectors) Split(bounds []Bounds) []Split {
	splits := make([]Split, 0, 8)
	for col := 0; col < v.Width; col++ {
		v.Sort(col)

		max, index := 0.0, 0
		mean, count := 0.0, 0.0
		for i := bounds[col].Begin; i < bounds[col].End; i++ {
			mean += v.Vectors[i].Vector[col]
			count++
		}
		mean /= count
		stddev := 0.0
		for i := bounds[col].Begin; i < bounds[col].End; i++ {
			diff := mean - v.Vectors[i].Vector[col]
			stddev += diff * diff
		}
		for i := bounds[col].Begin; i < bounds[col].End-1; i++ {
			meanA, meanB := 0.0, 0.0
			countA, countB := 0.0, 0.0
			for j := bounds[col].Begin; j < i+1; j++ {
				meanA += v.Vectors[j].Vector[col]
				countA++
			}
			for j := i + 1; j < bounds[col].End; j++ {
				meanB += v.Vectors[j].Vector[col]
				countB++
			}
			meanA /= countA
			meanB /= countB
			stddevA, stddevB := 0.0, 0.0
			for j := bounds[col].Begin; j < i+1; j++ {
				diff := meanA - v.Vectors[j].Vector[col]
				stddevA += diff * diff
			}
			for j := i + 1; j < bounds[col].End; j++ {
				diff := meanB - v.Vectors[j].Vector[col]
				stddevB += diff * diff
			}
			if v := stddev - (stddevA + stddevB); v > max {
				max, index = v, i
			}
		}
		splits = append(splits, Split{
			Col:   col,
			Index: index + 1,
			Var:   max,
		})
	}
	return splits
}

func (v *Vectors) SplitMulti(bounds []Bounds) []Split {
	splits := make([]Split, 0, 8)

	for col := 0; col < v.Width; col++ {
		v.Sort(col)

		vars := matrix.NewMatrix(v.Width, len(v.Vectors))
		for _, vector := range v.Vectors {
			for _, item := range vector.Vector {
				vars.Data = append(vars.Data, float32(item))
			}
		}

		multi := matrix.NewMultiFromData(vars.Slice(bounds[col].Begin, bounds[col].End).T())
		stddev := 0.0
		for j := 0; j < multi.E.Cols; j++ {
			x := float64(multi.E.Data[col*multi.E.Cols+j])
			stddev += x * x
		}
		stddev = math.Sqrt(stddev)

		max, index := 0.0, 0
		for i := bounds[col].Begin; i < bounds[col].End-1; i++ {
			multiA := matrix.NewMultiFromData(vars.Slice(bounds[col].Begin, i+1).T())
			multiB := matrix.NewMultiFromData(vars.Slice(i+1, bounds[col].End).T())
			stddevA, stddevB := 0.0, 0.0
			for j := 0; j < multiA.E.Cols; j++ {
				x := float64(multiA.E.Data[col*multiA.E.Cols+j])
				stddevA += x * x
			}
			stddevA = math.Sqrt(stddevA)
			for j := 0; j < multiB.E.Cols; j++ {
				x := float64(multiB.E.Data[col*multiB.E.Cols+j])
				stddevB += x * x
			}
			stddevB = math.Sqrt(stddevB)
			if v := stddev - (stddevA + stddevB); v > max {
				max, index = v, i
			}
		}
		splits = append(splits, Split{
			Col:   col,
			Index: index + 1,
			Var:   max,
		})
	}
	return splits
}

// K computes the K complexity
func (v *Vectors) K() int {
	cost := 0
	for col := 0; col < v.Width; col++ {
		v.Sort(col)
		labels := make([]uint8, 0, 8)
		for _, vector := range v.Vectors {
			labels = append(labels, vector.Labels...)
		}
		buffer := bytes.Buffer{}
		compress.Mark1Compress1(labels, &buffer)
		cost += buffer.Len()
	}
	return cost
}

// Vectors is a set of vectors
type Sorter struct {
	Vec *Vectors
	Col int
}

// Len is the length of Vectors
func (s Sorter) Len() int {
	return len(s.Vec.Vectors)
}

// Less is true of vector i is less than vector j
func (s Sorter) Less(i, j int) bool {
	return s.Vec.Vectors[i].Vector[s.Col] < s.Vec.Vectors[j].Vector[s.Col]
}

// Swap swaps two vectors
func (s Sorter) Swap(i, j int) {
	s.Vec.Vectors[i], s.Vec.Vectors[j] = s.Vec.Vectors[j], s.Vec.Vectors[i]
}

// Starlight is the starlight mode
func Starlight() {
	rng := matrix.Rand(1)

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	input := matrix.NewMatrix(4, 150)
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			input.Data = append(input.Data, float32(measure))
		}
	}
	synth := matrix.NewMultiFromData(input.T())
	synth.LearnA(&rng, nil)
	for i := 0; i < 150; i++ {
		vector := make([]float64, 4)
		measures := synth.Sample(&rng).Data
		for j := range vector {
			vector[j] = float64(measures[j])
		}
		datum.Fisher = append(datum.Fisher, iris.Iris{
			Label:    "Synth",
			Measures: vector,
		})
	}
	max := 0.0
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			if measure > max {
				max = measure
			}
		}
	}
	in := matrix.NewMatrix(4, 300)
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			in.Data = append(in.Data, float32(measure/max))
		}
	}
	input = in

	entropy := func(clusters []int) {
		ab, ba := [Clusters][Clusters]float64{}, [Clusters][Clusters]float64{}
		for i := range datum.Fisher {
			a := int(iris.Labels[datum.Fisher[i].Label])
			b := clusters[i]
			ab[a][b]++
			ba[b][a]++
		}
		entropy := 0.0
		for i := 0; i < Clusters; i++ {
			entropy += (1.0 / float64(Clusters)) * math.Log(1.0/float64(Clusters))
		}
		fmt.Println(-entropy, -(1.0/float64(Clusters))*math.Log(1.0/float64(Clusters)))
		for i := range ab {
			entropy := 0.0
			for _, value := range ab[i] {
				if value > 0 {
					p := value / 150
					entropy += p * math.Log(p)
				}
			}
			entropy = -entropy
			fmt.Println("ab", i, entropy)
		}
		for i := range ba {
			entropy := 0.0
			for _, value := range ba[i] {
				if value > 0 {
					p := value / 150
					entropy += p * math.Log(p)
				}
			}
			entropy = -entropy
			fmt.Println("ba", i, entropy)
		}
	}

	process := func(index int, sample matrix.Sample) Vectors {
		x1 := sample.Vars[0][0].Sample()
		y1 := sample.Vars[0][1].Sample()
		z1 := sample.Vars[0][2].Sample()
		w1 := x1.Add(y1.H(z1))

		x2 := sample.Vars[1][0].Sample()
		y2 := sample.Vars[1][1].Sample()
		z2 := sample.Vars[1][2].Sample()
		b1 := x2.Add(y2.H(z2))

		x3 := sample.Vars[2][0].Sample()
		y3 := sample.Vars[2][1].Sample()
		z3 := sample.Vars[2][2].Sample()
		w2 := x3.Add(y3.H(z3))

		x4 := sample.Vars[3][0].Sample()
		y4 := sample.Vars[3][1].Sample()
		z4 := sample.Vars[3][2].Sample()
		b2 := x4.Add(y4.H(z4))

		x5 := sample.Vars[4][0].Sample()
		y5 := sample.Vars[4][1].Sample()
		z5 := sample.Vars[4][2].Sample()
		w3 := x5.Add(y5.H(z5))

		x6 := sample.Vars[5][0].Sample()
		y6 := sample.Vars[5][1].Sample()
		z6 := sample.Vars[5][2].Sample()
		b3 := x6.Add(y6.H(z6))

		output := w3.MulT(w2.MulT(w1.MulT(input).Add(b1).Everett()).Add(b2).Everett()).Add(b3)

		vectors := Vectors{
			Width:   output.Cols,
			Vectors: make([]Vector, output.Rows),
			Rng:     rand.New(rand.NewSource(int64(index) + 1)),
		}
		for i := range vectors.Vectors {
			vector := make([]float64, vectors.Width)
			labels := make([]uint8, vectors.Width)
			for j := range vector {
				vector[j] = float64(output.Data[i*output.Cols+j])
			}
			vectors.Vectors[i] = Vector{
				Number: i,
				Vector: vector,
				Labels: labels,
			}
		}

		bounds := make([]Bounds, 0, 8)
		for i := 0; i < vectors.Width; i++ {
			bounds = append(bounds, Bounds{
				Begin: 0,
				End:   len(vectors.Vectors),
			})
		}
		splits := vectors.SplitMulti(bounds)
		boundsUpper := make([]Bounds, 0, 8)
		boundsLower := make([]Bounds, 0, 8)
		for i := range splits {
			boundsUpper = append(boundsUpper, Bounds{
				Begin: 0,
				End:   splits[i].Index,
			})
			boundsLower = append(boundsLower, Bounds{
				Begin: splits[i].Index,
				End:   len(vectors.Vectors),
			})
			for j := splits[i].Index; j < len(vectors.Vectors); j++ {
				vectors.Vectors[j].Labels[i] = 1
			}
		}
		splitsA := vectors.SplitMulti(boundsUpper)
		splitsB := vectors.SplitMulti(boundsLower)
		for i := range splitsA {
			//if splitsA[i].Var > splitsB[i].Var {
			for j := splitsA[i].Index; j < splits[i].Index; j++ {
				vectors.Vectors[j].Labels[i] = 2
			}
			//} else {
			for j := splitsB[i].Index; j < len(vectors.Vectors); j++ {
				vectors.Vectors[j].Labels[i] = 3
			}
			//}
		}
		return vectors
	}
	optimizer := matrix.NewOptimizer(&rng, 8, .1, 6, func(samples []matrix.Sample, x ...matrix.Matrix) {
		done := make(chan bool, 8)
		sample := func(index int, s *matrix.Sample) {
			vectors := process(index, *s)
			s.Cost = float64(vectors.K())
			done <- true
		}
		index, flight, cpus := 0, 0, runtime.NumCPU()
		for flight < cpus && index < len(samples) {
			go sample(index, &samples[index])
			index++
			flight++
		}
		for index < len(samples) {
			<-done
			flight--

			go sample(index, &samples[index])
			index++
			flight++
		}
		for i := 0; i < flight; i++ {
			<-done
		}
	}, matrix.NewCoord(4, 8), matrix.NewCoord(8, 1), matrix.NewCoord(16, 8), matrix.NewCoord(8, 1),
		matrix.NewCoord(16, 4), matrix.NewCoord(4, 1))
	var sample matrix.Sample
	for i := 0; i < 33; i++ {
		sample = optimizer.Iterate()
		fmt.Println(i, sample.Cost)
	}
	vectors := process(0, sample)
	sort.Slice(vectors.Vectors, func(i, j int) bool {
		return vectors.Vectors[i].Number < vectors.Vectors[j].Number
	})
	rawData := make([][]float64, len(vectors.Vectors))
	for i := range vectors.Vectors {
		rawData[i] = make([]float64, len(vectors.Vectors))
		for j := range rawData[i] {
			diff := 0.0
			for k, a := range vectors.Vectors[i].Labels {
				b := vectors.Vectors[j].Labels[k]
				if a != b {
					diff++
				}
			}
			rawData[i][j] = diff
		}
	}
	clusters, _, err := kmeans.Kmeans(1, rawData, Clusters, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i := range vectors.Vectors {
		fmt.Println(clusters[i], datum.Fisher[i].Label, vectors.Vectors[i].Labels)
	}
	entropy(clusters)
}

// Starlight2 is the starlight2 mode
func Starlight2() {
	rng := matrix.Rand(1)

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	input := matrix.NewMatrix(4, 150)
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			input.Data = append(input.Data, float32(measure))
		}
	}
	synth := matrix.NewMultiFromData(input.T())
	synth.LearnA(&rng, nil)
	for i := 0; i < 150; i++ {
		vector := make([]float64, 4)
		measures := synth.Sample(&rng).Data
		for j := range vector {
			vector[j] = float64(measures[j])
		}
		datum.Fisher = append(datum.Fisher, iris.Iris{
			Label:    "Synth",
			Measures: vector,
		})
	}
	max := 0.0
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			if measure > max {
				max = measure
			}
		}
	}
	in := matrix.NewMatrix(4, 300)
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			in.Data = append(in.Data, float32(measure/max))
		}
	}
	input = in

	entropy := func(clusters []int) {
		ab, ba := [Clusters][Clusters]float64{}, [Clusters][Clusters]float64{}
		for i := range datum.Fisher {
			a := int(iris.Labels[datum.Fisher[i].Label])
			b := clusters[i]
			ab[a][b]++
			ba[b][a]++
		}
		entropy := 0.0
		for i := 0; i < Clusters; i++ {
			entropy += (1.0 / float64(Clusters)) * math.Log(1.0/float64(Clusters))
		}
		fmt.Println(-entropy, -(1.0/float64(Clusters))*math.Log(1.0/float64(Clusters)))
		for i := range ab {
			entropy := 0.0
			for _, value := range ab[i] {
				if value > 0 {
					p := value / 150
					entropy += p * math.Log(p)
				}
			}
			entropy = -entropy
			fmt.Println("ab", i, entropy)
		}
		for i := range ba {
			entropy := 0.0
			for _, value := range ba[i] {
				if value > 0 {
					p := value / 150
					entropy += p * math.Log(p)
				}
			}
			entropy = -entropy
			fmt.Println("ba", i, entropy)
		}
	}

	process := func(index int, sample matrix.Sample) Vectors {
		x1 := sample.Vars[0][0].Sample()
		y1 := sample.Vars[0][1].Sample()
		z1 := sample.Vars[0][2].Sample()
		w1 := x1.Add(y1.H(z1))

		x2 := sample.Vars[1][0].Sample()
		y2 := sample.Vars[1][1].Sample()
		z2 := sample.Vars[1][2].Sample()
		w2 := x2.Add(y2.H(z2))

		x3 := sample.Vars[2][0].Sample()
		y3 := sample.Vars[2][1].Sample()
		z3 := sample.Vars[2][2].Sample()
		w3 := x3.Add(y3.H(z3))

		x4 := sample.Vars[3][0].Sample()
		y4 := sample.Vars[3][1].Sample()
		z4 := sample.Vars[3][2].Sample()
		w4 := x4.Add(y4.H(z4))

		x5 := sample.Vars[4][0].Sample()
		y5 := sample.Vars[4][1].Sample()
		z5 := sample.Vars[4][2].Sample()
		b1 := x5.Add(y5.H(z5))

		x6 := sample.Vars[5][0].Sample()
		y6 := sample.Vars[5][1].Sample()
		z6 := sample.Vars[5][2].Sample()
		w5 := x6.Add(y6.H(z6))

		x7 := sample.Vars[6][0].Sample()
		y7 := sample.Vars[6][1].Sample()
		z7 := sample.Vars[6][2].Sample()
		b2 := x7.Add(y7.H(z7))

		in := w4.MulT(input).Add(b1).Everett()
		q := w1.MulT(in)
		k := w2.MulT(in)
		v := w3.MulT(in)

		output := matrix.SelfAttention(q, k, v)
		output = w5.MulT(output).Add(b2)

		vectors := Vectors{
			Width:   output.Cols,
			Vectors: make([]Vector, output.Rows),
			Rng:     rand.New(rand.NewSource(int64(index) + 1)),
		}
		for i := range vectors.Vectors {
			vector := make([]float64, vectors.Width)
			labels := make([]uint8, vectors.Width)
			for j := range vector {
				vector[j] = float64(output.Data[i*output.Cols+j])
			}
			vectors.Vectors[i] = Vector{
				Number: i,
				Vector: vector,
				Labels: labels,
			}
		}

		bounds := make([]Bounds, 0, 8)
		for i := 0; i < vectors.Width; i++ {
			bounds = append(bounds, Bounds{
				Begin: 0,
				End:   len(vectors.Vectors),
			})
		}
		splits := vectors.Split(bounds)
		boundsUpper := make([]Bounds, 0, 8)
		boundsLower := make([]Bounds, 0, 8)
		for i := range splits {
			boundsUpper = append(boundsUpper, Bounds{
				Begin: 0,
				End:   splits[i].Index,
			})
			boundsLower = append(boundsLower, Bounds{
				Begin: splits[i].Index,
				End:   len(vectors.Vectors),
			})
			for j := splits[i].Index; j < len(vectors.Vectors); j++ {
				vectors.Vectors[j].Labels[i] = 1
			}
		}
		splitsA := vectors.Split(boundsUpper)
		splitsB := vectors.Split(boundsLower)
		for i := range splitsA {
			//if splitsA[i].Var > splitsB[i].Var {
			for j := splitsA[i].Index; j < splits[i].Index; j++ {
				vectors.Vectors[j].Labels[i] = 2
			}
			//} else {
			for j := splitsB[i].Index; j < len(vectors.Vectors); j++ {
				vectors.Vectors[j].Labels[i] = 3
			}
			//}
		}
		return vectors
	}
	optimizer := matrix.NewOptimizer(&rng, 8, .1, 7, func(samples []matrix.Sample, x ...matrix.Matrix) {
		done := make(chan bool, 8)
		sample := func(index int, s *matrix.Sample) {
			vectors := process(index, *s)
			s.Cost = float64(vectors.K())
			done <- true
		}
		index, flight, cpus := 0, 0, runtime.NumCPU()
		for flight < cpus && index < len(samples) {
			go sample(index, &samples[index])
			index++
			flight++
		}
		for index < len(samples) {
			<-done
			flight--

			go sample(index, &samples[index])
			index++
			flight++
		}
		for i := 0; i < flight; i++ {
			<-done
		}
	}, matrix.NewCoord(16, 16), matrix.NewCoord(16, 16), matrix.NewCoord(16, 16),
		matrix.NewCoord(4, 8), matrix.NewCoord(8, 1), matrix.NewCoord(16, 16), matrix.NewCoord(16, 1))
	var sample matrix.Sample
	for i := 0; i < 33; i++ {
		sample = optimizer.Iterate()
		fmt.Println(i, sample.Cost)
	}
	vectors := process(0, sample)
	sort.Slice(vectors.Vectors, func(i, j int) bool {
		return vectors.Vectors[i].Number < vectors.Vectors[j].Number
	})
	rawData := make([][]float64, len(vectors.Vectors))
	for i := range vectors.Vectors {
		rawData[i] = make([]float64, len(vectors.Vectors))
		for j := range rawData[i] {
			diff := 0.0
			for k, a := range vectors.Vectors[i].Labels {
				b := vectors.Vectors[j].Labels[k]
				if a != b {
					diff++
				}
			}
			rawData[i][j] = diff
		}
	}
	clusters, _, err := kmeans.Kmeans(1, rawData, Clusters, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i := range vectors.Vectors {
		fmt.Println(clusters[i], datum.Fisher[i].Label, vectors.Vectors[i].Labels)
	}
	entropy(clusters)
}

// Starlight3
func Starlight3() {
	rng := matrix.Rand(1)

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	input := matrix.NewMatrix(4, 150)
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			input.Data = append(input.Data, float32(measure))
		}
	}
	synth := matrix.NewMultiFromData(input.T())
	synth.LearnA(&rng, nil)
	for i := 0; i < 150; i++ {
		vector := make([]float64, 4)
		measures := synth.Sample(&rng).Data
		for j := range vector {
			vector[j] = float64(measures[j])
		}
		datum.Fisher = append(datum.Fisher, iris.Iris{
			Label:    "Synth",
			Measures: vector,
		})
	}
	max := 0.0
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			if measure > max {
				max = measure
			}
		}
	}
	in := matrix.NewMatrix(4, 300)
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			in.Data = append(in.Data, float32(measure/max))
		}
	}
	input = in

	process := func(index int, sample matrix.Sample) float64 {
		x1 := sample.Vars[0][0].Sample()
		y1 := sample.Vars[0][1].Sample()
		z1 := sample.Vars[0][2].Sample()
		selection := x1.Add(y1.H(z1))

		mean := make([]float64, selection.Cols)
		for i := 0; i < selection.Rows; i++ {
			for j := 0; j < selection.Cols; j++ {
				mean[j] += float64(input.Data[i*input.Cols+j])
			}
		}
		for i := range mean {
			mean[i] /= float64(selection.Rows)
		}
		stddev := make([]float64, selection.Cols)
		for i := 0; i < selection.Rows; i++ {
			for j := 0; j < selection.Cols; j++ {
				diff := mean[j] - float64(input.Data[i*input.Cols+j])
				stddev[j] += diff * diff
			}
		}

		vectors := make([]Vectors, selection.Cols)
		for i := range vectors {
			vectors[i] = Vectors{
				Width:   input.Cols,
				Vectors: make([]Vector, 0, input.Rows),
				Rng:     rand.New(rand.NewSource(int64(index) + 1)),
				Mean:    make([]float64, input.Cols),
				Stddev:  make([]float64, input.Cols),
			}
		}
		for i := 0; i < selection.Rows; i++ {
			for j := 0; j < selection.Cols; j++ {
				if selection.Data[i*selection.Cols+j] > 0 {
					vector := make([]float64, vectors[j].Width)
					labels := make([]uint8, vectors[j].Width)
					for j := range vector {
						vector[j] = float64(input.Data[i*input.Cols+j])
					}
					vectors[j].Vectors = append(vectors[j].Vectors, Vector{
						Number: i,
						Vector: vector,
						Labels: labels,
					})
				}
			}
		}

		for i := range vectors {
			for j := range vectors[i].Vectors {
				for k, value := range vectors[i].Vectors[j].Vector {
					vectors[i].Mean[k] += value
				}
			}
			for j := range vectors[i].Mean {
				vectors[i].Mean[j] /= float64(len(vectors[i].Vectors))
			}
			for j := range vectors[i].Vectors {
				for k, value := range vectors[i].Vectors[j].Vector {
					diff := vectors[i].Mean[k] - value
					vectors[i].Stddev[k] += diff * diff
				}
			}
			for j := range vectors[i].Stddev {
				vectors[i].Stddev[j] /= float64(len(vectors[i].Vectors))
			}
		}

		sum := 0.0
		for i, value := range stddev {
			sum += value
			for j := range vectors {
				sum -= vectors[j].Stddev[i]
			}
		}

		return sum
	}
	optimizer := matrix.NewOptimizer(&rng, 16, .1, 1, func(samples []matrix.Sample, x ...matrix.Matrix) {
		done := make(chan bool, 8)
		sample := func(index int, s *matrix.Sample) {
			s.Cost = process(index, *s)
			done <- true
		}
		index, flight, cpus := 0, 0, runtime.NumCPU()
		for flight < cpus && index < len(samples) {
			go sample(index, &samples[index])
			index++
			flight++
		}
		for index < len(samples) {
			<-done
			flight--

			go sample(index, &samples[index])
			index++
			flight++
		}
		for i := 0; i < flight; i++ {
			<-done
		}
	}, matrix.NewCoord(4, 300))
	var sample matrix.Sample
	for i := 0; i < 33; i++ {
		sample = optimizer.Iterate()
		fmt.Println(i, sample.Cost)
	}

	x1 := sample.Vars[0][0].Sample()
	y1 := sample.Vars[0][1].Sample()
	z1 := sample.Vars[0][2].Sample()
	selection := x1.Add(y1.H(z1))
	for i := 0; i < selection.Rows; i++ {
		index, max := 0, 0.0
		for j := 0; j < selection.Cols; j++ {
			if value := float64(selection.Data[i*selection.Cols+j]); value > max {
				index, max = j, value
			}
		}
		fmt.Println(index, datum.Fisher[i].Label)
	}
}

// Starlight4
func Starlight4() {
	rng := matrix.Rand(1)

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	max := 0.0
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			if measure > max {
				max = measure
			}
		}
	}
	input := matrix.NewMatrix(4, 150)
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			input.Data = append(input.Data, float32(measure/max))
		}
	}

	process := func(sample matrix.Sample) [][]float64 {
		x1 := sample.Vars[0][0].Sample()
		y1 := sample.Vars[0][1].Sample()
		z1 := sample.Vars[0][2].Sample()
		w1 := x1.Add(y1.H(z1))

		x2 := sample.Vars[1][0].Sample()
		y2 := sample.Vars[1][1].Sample()
		z2 := sample.Vars[1][2].Sample()
		b1 := x2.Add(y2.H(z2))

		x3 := sample.Vars[2][0].Sample()
		y3 := sample.Vars[2][1].Sample()
		z3 := sample.Vars[2][2].Sample()
		w2 := x3.Add(y3.H(z3))

		x4 := sample.Vars[3][0].Sample()
		y4 := sample.Vars[3][1].Sample()
		z4 := sample.Vars[3][2].Sample()
		b2 := x4.Add(y4.H(z4))

		output := w2.MulT(w1.MulT(input).Add(b1).Everett()).Add(b2)

		rawData := make([][]float64, output.Rows)
		for i := 0; i < output.Rows; i++ {
			for j := 0; j < output.Cols; j++ {
				rawData[i] = append(rawData[i], float64(output.Data[i*output.Cols+j]))
			}
		}
		meta := make([][]float64, output.Rows)
		for i := range meta {
			meta[i] = make([]float64, output.Rows)
		}

		for i := 0; i < 100; i++ {
			clusters, _, err := kmeans.Kmeans(int64(i+1), rawData, 3, kmeans.SquaredEuclideanDistance, -1)
			if err != nil {
				panic(err)
			}
			for i := range meta {
				target := clusters[i]
				for j, v := range clusters {
					if v == target {
						meta[i][j]++
					}
				}
			}
		}

		return meta
	}
	optimizer := matrix.NewOptimizer(&rng, 8, .1, 4, func(samples []matrix.Sample, x ...matrix.Matrix) {
		done := make(chan bool, 8)
		sample := func(s *matrix.Sample) {
			meta := process(*s)

			entropy := 0.0
			for i := range meta {
				sum := 0.0
				for _, value := range meta[i] {
					sum += value
				}
				if sum == 0 {
					continue
				}
				for _, value := range meta[i] {
					if value == 0 {
						continue
					}
					p := value / sum
					entropy += p * math.Log(p)
				}
			}
			s.Cost = -entropy / float64(len(meta))
			done <- true
		}
		index, flight, cpus := 0, 0, runtime.NumCPU()
		for flight < cpus && index < len(samples) {
			go sample(&samples[index])
			index++
			flight++
		}
		for index < len(samples) {
			<-done
			flight--

			go sample(&samples[index])
			index++
			flight++
		}
		for i := 0; i < flight; i++ {
			<-done
		}
	}, matrix.NewCoord(4, 8), matrix.NewCoord(8, 1), matrix.NewCoord(16, 16), matrix.NewCoord(16, 1))
	var sample matrix.Sample
	for i := 0; i < 33; i++ {
		sample = optimizer.Iterate()
		fmt.Println(i, sample.Cost)
	}

	meta := process(sample)
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, v := range clusters {
		fmt.Printf("%3d %15s %d\n", i, datum.Fisher[i].Label, v)
	}
}

var (
	// FlagOne
	FlagOne = flag.Bool("one", false, "one")
	// FlagTwo
	FlagTwo = flag.Bool("two", false, "two")
	// FlagThree
	FlagThree = flag.Bool("three", false, "three")
	// FlagFour
	FlagFour = flag.Bool("four", false, "four")
)

func main() {
	flag.Parse()

	if *FlagOne {
		Starlight()
		return
	} else if *FlagTwo {
		Starlight2()
		return
	} else if *FlagThree {
		Starlight3()
		return
	} else if *FlagFour {
		Starlight4()
		return
	}

	graph := pagerank.NewGraph64()

	graph.Link(1, 2, 1.0)
	graph.Link(2, 1, 1.0)
	graph.Link(2, 3, 1.0)
	graph.Link(3, 2, 1.0)
	graph.Link(3, 1, 1.0)
	graph.Link(1, 3, 1.0)

	graph.Link(4, 5, 1.0)
	graph.Link(5, 4, 1.0)
	graph.Link(5, 6, 1.0)
	graph.Link(6, 5, 1.0)
	graph.Link(6, 4, 1.0)
	graph.Link(4, 6, 1.0)

	graph.Link(1, 4, 1.0)
	graph.Link(4, 1, 1.0)

	graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
		fmt.Println("Node", node, "has a rank of", rank)
	})

	previous := make([][]float64, 4)
	previous[0] = []float64{0, 0}
	previous[1] = []float64{1, 0}
	previous[2] = []float64{1, 1}
	previous[3] = []float64{0, 1}

	for i := 0; i < 8; i++ {
		meta := make([][]float64, 4)
		for j := range previous {
			for k := range previous {
				sum := 0.0
				for l := range previous[k] {
					diff := previous[j][l] - previous[k][l]
					sum += diff * diff
				}
				meta[j] = append(meta[j], math.Sqrt(sum))
			}
		}
		for j := range meta {
			fmt.Println(meta[j])
		}
		fmt.Println()
		previous = meta
	}
}
