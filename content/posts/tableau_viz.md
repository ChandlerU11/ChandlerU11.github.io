---
title: Running Data Dashboard
date: 2024-08-14
author: Chandler Underwood
description: Using the running data I cleaned up in a previous post, I create a Tableau dashboard that allows for dynamic granularity changes from the year down to day level.
ShowToc: true
TocOpen: true
---

{{< rawhtml >}}
<!DOCTYPE html>
<html>
<head>
<script async src="https://www.googletagmanager.com/gtag/js?id=G-0NTZD30YVX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-0NTZD30YVX');
</script>
</head>
</html>
{{< /rawhtml >}}


# About
This post contains the dashboard I built to go along with my [data cleaning project]({{< ref "clean_data.md" >}}) where I clean up my own running data from college. Shoutout to Andy Kriebel and his awesome YouTube [video](https://www.youtube.com/watch?v=EZMLjMaZYSs&t=308s) for getting me started! 

The dashboard allows for dynamic exploration of the time series data from years down to days. I would recommend opening up the visualization to full screen, so you can see the "timeframe snapshots" along with my most important running stats.


# The Dashboard
{{< rawhtml >}}
<!DOCTYPE html>
<html>
<head>
<div class='tableauPlaceholder' id='viz1709151872311' style='position: relative'><noscript><a href='#'><img alt='Running Data DashboardThis time series dashboard allows for year, month, week  drill down. To drill down click on the bar for the desired timeframe. To drill up press ctrl + z or &quot;Undo my last action&quot; at the bottom of the dasboard. The bar charts showcas ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ru&#47;RunningDataDashboard_17065579590940&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='RunningDataDashboard_17065579590940&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ru&#47;RunningDataDashboard_17065579590940&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1709151872311');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='2427px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
</body>
</html>

{{< /rawhtml >}}

