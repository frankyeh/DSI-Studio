param(
    [Parameter(Mandatory,Position=0)]
    [string]$Agent = 'Claude',

    [Parameter(Position=1)]
    [string]$Session,

    [Parameter(Mandatory,Position=2)]
    [string]$Target,

    [Parameter(Position=3,ValueFromRemainingArguments)]
    [string[]]$Value,

    [string]$Chat
)

if(!$Session -and $Agent -ieq 'Codex')
{
    $Session = $env:CODEX_THREAD_ID
}
if(!$Session)
{
    throw 'Missing Session. Codex must use CODEX_THREAD_ID.'
}

function Convert-DsiValue([string]$Text)
{
    $integer = 0L
    if([long]::TryParse($Text,[ref]$integer))
    {
        return $integer
    }
    $number = 0.0
    if([double]::TryParse($Text,
        [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref]$number))
    {
        return $number
    }
    return $Text
}

$request = [ordered]@{agent=$Agent; session=$Session}
switch($Target.ToUpper())
{
    'LIST'  {$request.request = 'LIST'}
    'LOG'   {$request.request = 'LOG'}
    'CHAT'  {$request.request = 'CHAT';  $request.chat = $Value -join ' '}
    'TITLE' {$request.request = 'TITLE'; $request.title = $Value -join ' '}
    default
    {
        if(!$Value.Count)
        {
            throw 'Missing command name.'
        }
        $request.request = 'CMD'
        $request.window = $Target
        $request.command = [ordered]@{cmd=$Value[0]}
        $param = @($Value | Select-Object -Skip 1 | ForEach-Object {Convert-DsiValue $_})
        if($param.Count -eq 1)
        {
            $request.command.param = $param[0]
        }
        elseif($param.Count -gt 1)
        {
            $request.command.param = $param
        }
    }
}
if($Chat)
{
    $request.chat = $Chat
}

$pipe = $writer = $reader = $null
try
{
    $pipe = [IO.Pipes.NamedPipeClientStream]::new('.','dsi-studio')
    $pipe.Connect(5000)
    $utf8 = [Text.UTF8Encoding]::new($false)
    $writer = [IO.StreamWriter]::new($pipe,$utf8,1024,$true)
    $reader = [IO.StreamReader]::new($pipe,$utf8,$false,1024,$true)
    $writer.Write(($request | ConvertTo-Json -Compress -Depth 8))
    $writer.Flush()
    $reader.ReadToEnd()
}
finally
{
    foreach($stream in @($reader,$writer,$pipe))
    {
        if($stream)
        {
            try {$stream.Dispose()} catch [IO.IOException] {}
        }
    }
}
