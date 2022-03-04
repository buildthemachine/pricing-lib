# pricing-lib

This is the quantitative financial derivative pricing library currently under development. Before working on the files, make sure to: 

<ol>
    <li>Set your PYTHONPATH env variable such that it includes the absolute directory of pricing-lib</li>
    <li>Sync the packages with requirements.txt</li>
</ol>

The products supported currently include:  
<ol>
    <li>Vanilla European, American and Bermudan options. The undelrying volatility dynamics can be BS, CEV, or SABR.
    <li>Barrier options, digital options
</ol>

It also supports numerical PDE, and Monte Carlo engines for pricing. 