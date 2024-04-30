// Copyright 2024 The Starlight Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"

	"github.com/pointlander/compress"
	"github.com/pointlander/datum/iris"
	"github.com/pointlander/matrix"
	"github.com/pointlander/starlight/kmeans"
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

	entropy := func(clusters []int) {
		ab, ba := [3][3]float64{}, [3][3]float64{}
		for i := range datum.Fisher {
			a := int(iris.Labels[datum.Fisher[i].Label])
			b := clusters[i]
			ab[a][b]++
			ba[b][a]++
		}
		entropy := 0.0
		for i := 0; i < 3; i++ {
			entropy += (1.0 / 3.0) * math.Log(1.0/3.0)
		}
		fmt.Println(-entropy, -(1.0/3.0)*math.Log(1.0/3.0))
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

		output := w2.MulT(w1.MulT(input).Add(b1).Everett()).Add(b2)

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
			if splitsA[i].Var > splitsB[i].Var {
				for j := splitsA[i].Index; j < splits[i].Index; j++ {
					vectors.Vectors[j].Labels[i] = 2
				}
			} else {
				for j := splitsB[i].Index; j < len(vectors.Vectors); j++ {
					vectors.Vectors[j].Labels[i] = 2
				}
			}
		}
		return vectors
	}
	optimizer := matrix.NewOptimizer(&rng, 8, .1, 4, func(samples []matrix.Sample, x ...matrix.Matrix) {
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
	}, matrix.NewCoord(4, 8), matrix.NewCoord(8, 1), matrix.NewCoord(16, 3), matrix.NewCoord(3, 1))
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
	clusters, _, err := kmeans.Kmeans(1, rawData, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i := range vectors.Vectors {
		fmt.Println(clusters[i], datum.Fisher[i].Label, vectors.Vectors[i].Labels)
	}
	entropy(clusters)
}

func main() {
	Starlight()
}
