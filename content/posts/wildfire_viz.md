---
title: (Under Construction) Visualizing ​28 Years of U.S. Wildfires
date: 2024-05-17
author: Chandler Underwood
description: This post is not complete yet, but in this project I build two different visualizations to portray the issue of increasing wildfires in the U.S. 
ShowToc: true
TocOpen: true
---

## Acreage Burn Percent Change Choropleth Map 

Below is a Choropleth Map that shows percent change in acreage burn by U.S. county. The percent change for a county is found by calculating the mean acreage burn for a county for the two date ranges 1992-2006 (**Old**) and 2007-2020 (**New**), subtracting them, and multiplying the result by 100. Counties that did not report wildire burn acreage are represented by the median burn acreage percent change, +3%.

**Percent Change = (New - Old) / Old x 100**

{{< rawhtml >}}
<html>
<head>

<meta charset="utf-8">    <div id="observablehq-key-82a06465"></div>
    <div id="observablehq-chart-82a06465"></div>
<p>Data Source: <a href="https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6">USDA Research Data Archive</a></p><p>

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@observablehq/inspector@5/dist/inspector.css">
<script type="module">
import {Runtime, Inspector} from "https://cdn.jsdelivr.net/npm/@observablehq/runtime@5/dist/runtime.js";
import define from "https://api.observablehq.com/d/c97b212669e37256.js?v=3";
new Runtime().module(define, name => {
if (name === "key") return new Inspector(document.querySelector("#observablehq-key-82a06465"));
if (name === "chart") return new Inspector(document.querySelector("#observablehq-chart-82a06465"));
});
</script>
 
</script>
</head>
</html>
{{< /rawhtml >}}

The dataset had several outliers with the largest having a percent increase of over 1,000,000% (Pondera County, Montana).
However, most fell between -29% decrease and 48% increase in acreage burn. The data is binned into colors
based on its sign and the quantile it belongs to. Negative numbers are colored blue while
positive numbers up to the 75th percentile are colored yellow. I accounted for outliers by
creating extra bins that grow exponentially. After the 1,000% bin, we multiply the bin value by
four until we reach 1,024,000%.

## U.S. Total Acreage Burn Heat Map
{{< rawhtml >}}
<!DOCTYPE html>
<meta charset="utf-8">

<!-- Load d3.js -->
<script src="https://d3js.org/d3.v4.js"></script>

<!-- Create a div where the graph will take place -->
<div id="my_dataviz" style="background-color: #e6e6e4;"></div>
<script>
// set the dimensions and margins of the graph
var margin = {top: 30, right: 35, bottom: 35, left: 65},
width = 750 - margin.left - margin.right,
height = 275 - margin.top - margin.bottom;
// append the svg object to the body of the page
var my_dataviz = d3.select("#my_dataviz")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");
// Labels of row and columns
var myGroups = ["1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
var myVars = ['Janurary','February', 'March','April','May', 'June','July','August','September','October','November','December']
// Build X scales and axis:
var x = d3.scaleBand()
.range([ 0, width - 100])
.domain(myGroups)
.padding(0.01);
my_dataviz.append("g")
.attr("transform", "translate(0," + height + ")")
.call(d3.axisBottom(x))
.selectAll("text")  
      .style("text-anchor", "end")
      .attr("dx", "-1em")
      .attr("dy", "-0.1em")
      .attr("transform", "rotate(-60)");
// Build X scales and axis:
var y = d3.scaleBand()
.range([ height, 0 ])
.domain(myVars)
.padding(0.01);
my_dataviz.append("g")
.call(d3.axisLeft(y));
// Build color scale
var myColor = d3.scaleLinear()
    .range(["#fdd49e","#fdbb84","#fc8d59","#e34a33", "#b30000", "#636363", "#252525"])
    .domain([7000, 76000, 200000, 560000, 1200000, 3500000, 5250000])
// create a list of keys
var keys = [7000, 76000, 200000, 560000, 1200000, 3500000, 5250000]
// Add one dot in the legend for each name.
var size = 10
my_dataviz.selectAll("mydots")
    .data(keys)
    .enter()
    .append("rect")
    .attr("x", 565)
    .attr("y", function(d,i){ return 50 + i*(size+5)}) // 100 is where the first dot appears. 25 is the distance between dots
    .attr("width", size)
    .attr("height", size)
    .style("fill", function(d){ return myColor(d)})
// Add one dot in the legend for each name.
my_dataviz.selectAll("mylabels")
    .data(keys)
    .enter()
    .append("text")
    .attr("title", "Temp")
    .attr("x", 565 + size*1.2)
    .attr("y", function(d,i){ return 50 + i*(size+5) + (size/2)}) // 100 is where the first dot appears. 25 is the distance between dots
    //.style("fill", function(d){ return color(d)})
    .text(function(d){ return d})
    .attr("text-anchor", "left")
    .style("alignment-baseline", "middle")
d3.csv("https://raw.githubusercontent.com/ChandlerU11/temp_data/main/heat_map.csv", function(data) {
// create a tooltip
var tooltip = d3.select("#my_dataviz")
    .append("div")
    .style("opacity", 0)
    .attr("class", "tooltip")
    .style("background-color", "white")
    .style("border", "solid")
    .style("border-width", "2px")
    .style("border-radius", "5px")
    .style("padding", "5px")
    .style("width", "400px")
// Three function that change the tooltip when user hover / move / leave a cell
var mouseover = function(d) {
    tooltip.style("opacity", 1)
}
var mousemove = function(d) {
    tooltip
    .html("Acreage Burned in " + d.month_name + ", " + d.years + ": " + d.size)
    .style("left", (d3.mouse(this)[0]+70) + "px")
    .style("top", (d3.mouse(this)[1]) + "px")
}
var mouseleave = function(d) {
    tooltip.style("opacity", 0)
}
// add the squares
my_dataviz.selectAll()
    .data(data, function(d) {return d.years+':'+d.month_name;})
    .enter()
    .append("rect")
    .attr("x", function(d) { return x(d.years) })
    .attr("y", function(d) { return y(d.month_name) })
    .attr("width", x.bandwidth() )
    .attr("height", y.bandwidth() )
    .style("fill", function(d) { return myColor(d.size)} )
    .on("mouseover", mouseover)
    .on("mousemove", mousemove)
    .on("mouseleave", mouseleave)
})

<!-- ----------------->
</script>
{{< /rawhtml >}}

## Some High-level Stats

- **64%** of U.S. Counties have seen an increase in the amount of land burned by wildfires when compared their averages of the 1990's and early 2000's.

- **1.67 F** That's how many degrees hotter the year 2020 was when compared to an average year in the 1990's.

## Data Issues
I encountered a couple data related issues when working on this project. In
the USDA Dataset, some counties did not report wildfire burn acreage in the 90’s. This made
it impossible to calculate the percentage change for those counties. To fix this I used the
median value (~3%) for percent change to fill in the NULL values in the table following the
percent change calculation. The majority of the counties were in several states across the
Midwest and Alaska.

There were also some counties that never reported wildfire acreage in the 90’s or in more
recent years, so their FIPS codes (used to identify counties in the choropleth) were not present in the final dataset.
